"""
Consistency Scorer using Perplexity Delta
==========================================
Implements the Generative Reasoning approach for narrative consistency detection.

Core Idea:
----------
If a backstory is CONSISTENT with a novel, then conditioning the language model
on that backstory should REDUCE its perplexity when predicting novel text.

The perplexity delta is computed as:
    Delta = Loss(novel_chunk | empty) - Loss(novel_chunk | backstory)

Interpretation:
- Positive delta: Backstory helps predict novel -> CONSISTENT
- Negative delta: Backstory hurts prediction -> CONTRADICTORY
- Zero delta: Backstory provides no information -> NEUTRAL
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np

import torch
import torch.nn as nn
from tokenizers import Tokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.textpath import TextPath


class ConsistencyScorer:
    """
    Score narrative consistency using Perplexity Delta.
    
    This replaces the MLP classification head with a principled
    information-theoretic approach based on conditional language modeling.
    """
    
    def __init__(
        self,
        model: TextPath,
        tokenizer: Tokenizer,
        device: str = 'cpu',
        max_context_len: int = 256,
        max_target_len: int = 256,
    ):
        """
        Initialize the consistency scorer.
        
        Args:
            model: Pretrained TextPath model (in LM mode, not classification mode)
            tokenizer: Tokenizer matching the model vocabulary
            device: Device to run inference on
            max_context_len: Maximum tokens for backstory/context
            max_target_len: Maximum tokens for novel chunk/target
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len
        
        # Ensure model is in eval mode
        self.model.eval()
        self.model.to(device)
        
        # Get special token IDs
        self.pad_id = tokenizer.token_to_id('[PAD]') or 0
        
        print(f"ConsistencyScorer initialized on {device}")
    
    def _tokenize(self, text: str, max_len: int) -> torch.Tensor:
        """Tokenize text and return tensor."""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:max_len]
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Robustly retrieve embeddings from the BDH model.
        Checks multiple common attribute names and falls back to module search.
        """
        bdh = self.model.bdh
        
        # 1. Try common attribute names
        if hasattr(bdh, 'token_embedding_table'):
            return bdh.token_embedding_table(input_ids)
        if hasattr(bdh, 'wte'):
            return bdh.wte(input_ids)
        if hasattr(bdh, 'emb'):
            return bdh.emb(input_ids)
        if hasattr(bdh, 'encoder') and isinstance(bdh.encoder, nn.Embedding):
            return bdh.encoder(input_ids)
            
        # 2. Fallback: Search for the first nn.Embedding module
        # This handles cases where the embedding layer is buried in a submodule
        for name, module in bdh.named_modules():
            if isinstance(module, nn.Embedding):
                # Verify vocab size match to be safe
                if module.num_embeddings == self.model.config.vocab_size:
                    return module(input_ids)
        
        # 3. If no embedding layer found (unlikely for LM), raise error
        raise AttributeError(
            f"Could not find embedding layer in BDH model. Checked: token_embedding_table, wte, emb, encoder."
        )

    def compute_delta(
        self,
        backstory: str,
        novel_chunk: str
    ) -> Tuple[float, float]:
        """
        Compute perplexity delta for consistency scoring.
        
        Delta = Loss(novel_chunk | empty) - Loss(novel_chunk | backstory)
        
        Args:
            backstory: The backstory/claim to check for consistency
            novel_chunk: Retrieved chunk from the novel (evidence)
            
        Returns:
            Tuple of (delta, retrieval_score):
            - delta: Positive = consistent, negative = contradictory
            - retrieval_score: Cosine similarity between backstory and chunk embeddings
        """
        # Tokenize inputs
        backstory_ids = self._tokenize(backstory, self.max_context_len)
        novel_ids = self._tokenize(novel_chunk, self.max_target_len)
        
        # Empty context for baseline
        empty_context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        # Compute baseline loss (no context)
        loss_base = self.model.calculate_conditional_loss(empty_context, novel_ids)
        
        # Compute conditioned loss (with backstory context)
        loss_conditioned = self.model.calculate_conditional_loss(backstory_ids, novel_ids)
        
        # Delta: positive means backstory helps predict novel (consistent)
        delta = loss_base - loss_conditioned
        
        # Compute embedding-based similarity as additional feature
        cosine_sim = self._compute_embedding_similarity(backstory_ids, novel_ids)
        
        return delta, cosine_sim
    
    def _compute_embedding_similarity(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between context and target embeddings.
        Uses the model's embedding layer for representation.
        """
        with torch.no_grad():
            # Get embeddings from the model using robust helper
            context_emb = self._get_embeddings(context_ids)  # (1, L1, D)
            target_emb = self._get_embeddings(target_ids)    # (1, L2, D)
            
            # Mean pool over sequence dimension
            context_vec = context_emb.mean(dim=1)  # (1, D)
            target_vec = target_emb.mean(dim=1)    # (1, D)
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(context_vec, target_vec)
            
            return cos_sim.item()
    
    def score_sample(
        self,
        backstory: str,
        novel_chunks: List[str],
        retrieval_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Score a single sample using multiple retrieved chunks.
        
        Args:
            backstory: The backstory/claim to check
            novel_chunks: List of retrieved chunks from the novel
            retrieval_scores: Optional retrieval similarity scores
            
        Returns:
            Dictionary with scoring features:
            - delta_mean: Mean delta across chunks
            - delta_max: Maximum delta (most helpful chunk)
            - cosine_mean: Mean embedding similarity
            - retrieval_score: Mean retrieval score (if provided)
        """
        deltas = []
        cosines = []
        
        for chunk in novel_chunks:
            if chunk.strip():  # Skip empty chunks
                delta, cosine = self.compute_delta(backstory, chunk)
                deltas.append(delta)
                cosines.append(cosine)
        
        if not deltas:
            return {
                'delta_mean': 0.0,
                'delta_max': 0.0,
                'cosine_mean': 0.0,
                'retrieval_score': 0.0
            }
        
        result = {
            'delta_mean': np.mean(deltas),
            'delta_max': np.max(deltas),
            'cosine_mean': np.mean(cosines),
        }
        
        if retrieval_scores:
            result['retrieval_score'] = np.mean(retrieval_scores)
        else:
            result['retrieval_score'] = 0.0
        
        return result
    
    def get_features(
        self,
        backstory: str,
        novel_chunks: List[str],
        retrieval_scores: Optional[List[float]] = None
    ) -> List[float]:
        """
        Get feature vector for calibration model.
        
        Returns:
            List of features: [delta_mean, delta_max, cosine_mean, retrieval_score]
        """
        scores = self.score_sample(backstory, novel_chunks, retrieval_scores)
        return [
            scores['delta_mean'],
            scores['delta_max'],
            scores['cosine_mean'],
            scores['retrieval_score']
        ]


def load_scorer(
    model_path: str,
    tokenizer_path: str,
    device: str = 'cpu'
) -> ConsistencyScorer:
    """
    Load a ConsistencyScorer from saved model checkpoint.
    
    Args:
        model_path: Path to the pretrained TextPath model checkpoint
        tokenizer_path: Path to the tokenizer JSON file
        device: Device to load model on
        
    Returns:
        Initialized ConsistencyScorer
    """
    from src.models.textpath import TextPath, TextPathConfig
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    if config is None:
        # Fallback to default config
        config = TextPathConfig(
            vocab_size=tokenizer.get_vocab_size(),
            classification_mode=False
        )
    else:
        # Ensure classification mode is off
        config.classification_mode = False
    
    # Initialize model
    model = TextPath(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return ConsistencyScorer(model, tokenizer, device)


# ==============================================================================
# Testing
# ==============================================================================

def test_consistency_scorer():
    """Test the consistency scorer with dummy data."""
    print("=" * 60)
    print("TESTING CONSISTENCY SCORER")
    print("=" * 60)
    
    from src.models.textpath import TextPath, TextPathConfig
    
    # Create small test model
    config = TextPathConfig(
        vocab_size=1000,
        max_seq_len=128,
        n_heads=4,
        n_neurons=256,
        d_model=64,
        n_layers=2,
        classification_mode=False
    )
    
    model = TextPath(config)
    
    # Create dummy tokenizer-like object for testing
    class DummyTokenizer:
        def encode(self, text):
            class Encoding:
                ids = [i % 1000 for i in range(len(text.split()))]
            return Encoding()
        
        def token_to_id(self, token):
            return 0
    
    tokenizer = DummyTokenizer()
    
    # Create scorer
    scorer = ConsistencyScorer(model, tokenizer, device='cpu')
    
    # Test scoring
    backstory = "Edmond was imprisoned in the Chateau dIf for many years."
    novel_chunk = "The prisoner spent his days in darkness, dreaming of escape."
    
    delta, cosine = scorer.compute_delta(backstory, novel_chunk)
    print(f"\nTest sample:")
    print(f"  Backstory: {backstory[:50]}...")
    print(f"  Novel chunk: {novel_chunk[:50]}...")
    print(f"  Delta: {delta:.4f}")
    print(f"  Cosine similarity: {cosine:.4f}")
    
    # Test feature extraction
    features = scorer.get_features(backstory, [novel_chunk])
    print(f"\nFeature vector: {features}")
    
    print("\nConsistency scorer test complete!")


if __name__ == "__main__":
    test_consistency_scorer()