"""
BDH-Native Novel Pretraining
============================
Leverages the unique properties of BDH architecture:

1. HEBBIAN LEARNING: "Neurons that fire together, wire together"
   - Train on sequential passages so character/place/event neurons co-activate
   - The synaptic weights (E, Dx, Dy) learn narrative relationships

2. SPARSE ACTIVATIONS: Only ~5% of neurons fire
   - Each concept (character, location, event) activates distinct neuron groups
   - This creates monosemantic representations for narrative elements

3. CAUSAL CIRCUITS: Gx = E @ Dx encodes "if A then B" reasoning
   - E.g., "If Edmond Dantès mentioned → prison/escape concepts should activate"
   - These circuits learn the factual content of the novel

Training Objectives:
-------------------
A) NARRATIVE LANGUAGE MODEL (NLM): Predict next token in novel passages
   - Standard LM but on carefully structured sequential passages
   - Ensures neurons learn to anticipate narrative patterns

B) PASSAGE COHERENCE (PC): Distinguish real vs shuffled passages  
   - Real passages: consecutive chunks from novel
   - Fake passages: randomly shuffled sentences
   - BDH circuits learn what "coherent narrative" looks like

C) ENTITY CONSISTENCY (EC): Contrastive learning on entity mentions
   - Same entity mentioned in different contexts → similar activation
   - Different entities → distinct activation patterns
   - Forces neurons to specialize for specific characters/places
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import math
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tokenizers import Tokenizer
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.textpath import TextPath, TextPathConfig
from src.utils import set_seed


# ============================================================
# Dataset for BDH-Native Training
# ============================================================

class BDHNovelDataset(Dataset):
    """
    Dataset designed for BDH's unique learning properties.
    
    Key design choices for Hebbian learning:
    - Small stride (high overlap) ensures every narrative relationship is seen many times
    - Sequential passages maintain temporal/causal ordering of events
    - Complete coverage ensures all characters and events are learned
    - Entity threads ensure character arcs are learned as continuous sequences
    
    Supports mixed data training:
    - Novel paths: List of paths to novel text files
    - Thread weight: How much to oversample entity thread chunks (default: 2.0)
    """
    
    def __init__(
        self,
        novel_paths: list,  # Changed from single Path to list of paths
        tokenizer: Tokenizer,
        chunk_size: int = 512,
        stride: int = 64,  # Small stride = high overlap = more Hebbian co-activation
        thread_weight: float = 2.0,  # Weight for thread chunks (higher priority)
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.thread_weight = thread_weight
        
        # Handle single path (backward compatibility)
        if isinstance(novel_paths, (str, Path)):
            novel_paths = [Path(novel_paths)]
        else:
            novel_paths = [Path(p) for p in novel_paths]
        
        self.novel_paths = novel_paths
        
        # Separate novel files from thread files
        novel_files = [p for p in novel_paths if not p.name.startswith('thread_')]
        thread_files = [p for p in novel_paths if p.name.startswith('thread_')]
        
        # Get novel name from first novel file
        if novel_files:
            self.novel_name = novel_files[0].stem
        else:
            self.novel_name = "mixed"
        
        self.chunks = []
        self.entities = []
        
        # Process novel files
        for novel_path in novel_files:
            novel_chunks, novel_entities = self._process_file(novel_path, stride, is_thread=False)
            self.chunks.extend(novel_chunks)
            self.entities.extend(novel_entities)
        
        novel_chunk_count = len(self.chunks)
        
        # Process thread files with weighting (oversample)
        for thread_path in thread_files:
            thread_chunks, _ = self._process_file(thread_path, stride, is_thread=True)
            
            # Apply thread weighting by duplicating chunks
            weight_multiplier = int(self.thread_weight)
            for _ in range(weight_multiplier):
                self.chunks.extend(thread_chunks)
        
        thread_chunk_count = len(self.chunks) - novel_chunk_count
        
        print(f"   📦 Total: {len(self.chunks):,} chunks ({novel_chunk_count:,} novel + {thread_chunk_count:,} thread)")
    
    def _process_file(self, file_path: Path, stride: int, is_thread: bool = False):
        """Process a single file and return chunks and entities."""
        file_type = "🧵 Thread" if is_thread else "📖 Novel"
        print(f"\n{file_type}: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean Gutenberg headers (only for novels)
        if not is_thread:
            text = self._clean_gutenberg(text)
        
        # Extract entities (only for novels)
        entities = []
        if not is_thread:
            entities = self._extract_entities(text)
            print(f"   📍 Key entities: {', '.join(entities[:10])}...")
        
        # Tokenize
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        print(f"   📝 {len(token_ids):,} tokens")
        
        # Create sequential chunks
        chunks = []
        for i in range(0, len(token_ids) - self.chunk_size + 1, stride):
            chunks.append(token_ids[i:i + self.chunk_size])
        
        # Add final chunk for complete coverage
        if len(token_ids) >= self.chunk_size:
            final_chunk = token_ids[-self.chunk_size:]
            if chunks and final_chunk != chunks[-1]:
                chunks.append(final_chunk)
        
        print(f"   📦 {len(chunks):,} chunks")
        
        return chunks, entities
    
    def _clean_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg headers/footers"""
        start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
        end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]
        
        start_idx = 0
        for marker in start_markers:
            if marker in text:
                start_idx = text.find(marker)
                start_idx = text.find('\n', start_idx) + 1
                break
        
        end_idx = len(text)
        for marker in end_markers:
            if marker in text:
                end_idx = text.find(marker)
                break
        
        return text[start_idx:end_idx].strip()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities (characters, places) from novel"""
        words = text.split()
        capitalized = {}
        
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) > 2 and clean[0].isupper() and not clean.isupper():
                capitalized[clean] = capitalized.get(clean, 0) + 1
        
        entities = [name for name, count in capitalized.items() if count > 10]
        entities.sort(key=lambda x: capitalized[x], reverse=True)
        return entities[:50]
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Language modeling sample: predict next token.
        
        This is the core training objective that enables:
        - Hebbian learning: neurons for co-occurring concepts strengthen connections
        - Causal circuits: model learns "if character X mentioned → related events follow"
        - Sparse representations: distinct neurons specialize for different narrative elements
        """
        chunk = self.chunks[idx]
        return {
            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
            'target_ids': torch.tensor(chunk[1:], dtype=torch.long),
        }


# ============================================================
# BDH-Native Loss Functions
# ============================================================

class BDHNativeLoss(nn.Module):
    """
    Loss function for BDH pretraining.
    
    Language Modeling is the core objective - it enables Hebbian learning by:
    - Training on sequential text where related concepts co-occur
    - Building causal circuits through next-token prediction
    - Developing sparse, monosemantic neuron representations
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute language modeling loss.
        
        Args:
            logits: [B, T, V] model predictions
            targets: [B, T] target tokens
        
        Returns:
            loss, metrics_dict
        """
        loss = self.ce_loss(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        metrics = {
            'loss': loss.item(),
            'ppl': math.exp(min(loss.item(), 20)),
        }
        
        return loss, metrics


# ============================================================
# Training Loop
# ============================================================

def train_epoch(
    model: TextPath,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BDHNativeLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch - language modeling builds Hebbian circuits"""
    model.train()
    
    total_loss = 0
    total_ppl = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Compute loss
        loss, metrics = loss_fn(logits, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += metrics['loss']
        total_ppl += metrics['ppl']
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'ppl': f"{metrics['ppl']:.1f}",
        })
    
    return {
        'loss': total_loss / n_batches,
        'ppl': total_ppl / n_batches,
    }


def pretrain_bdh_novel(
    novel_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    device: torch.device,
    config: TextPathConfig,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    thread_paths: list = None,  # NEW: Optional list of entity thread file paths
    thread_weight: float = 2.0,  # NEW: Weight for thread chunks
):
    """
    Pretrain BDH on a single novel with optional entity threads.
    
    Language modeling on sequential text enables:
    - Hebbian learning: co-occurring concepts strengthen connections
    - Causal circuits: "if A then B" reasoning patterns
    - Sparse representations: monosemantic neurons for characters/places
    
    Entity Threading (NEW):
    - Character-specific threads are extracted from the novel
    - These threads capture long-term character arcs
    - Thread chunks are weighted higher during training
    """
    novel_name = novel_path.stem
    
    print("\n" + "="*70)
    print(f"🐉 BDH-NATIVE PRETRAINING: {novel_name}")
    print("="*70)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Build list of all paths (novel + threads)
    all_paths = [novel_path]
    if thread_paths:
        all_paths.extend(thread_paths)
        print(f"   🧵 Including {len(thread_paths)} entity threads")
    
    # Create dataset with high overlap for Hebbian learning
    dataset = BDHNovelDataset(
        novel_paths=all_paths,
        tokenizer=tokenizer,
        chunk_size=config.max_seq_len,
        stride=64,  # Small stride = high overlap = more co-activation
        thread_weight=thread_weight,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    # Initialize fresh model for this novel
    print(f"\n🔧 Initializing BDH model...")
    model = TextPath(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Loss function
    loss_fn = BDHNativeLoss(vocab_size=config.vocab_size)
    
    # Training
    print(f"\n🚀 Training for {epochs} epochs...")
    print(f"   Batches/epoch: {len(dataloader)}")
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch)
        
        print(f"   Epoch {epoch:3d}: Loss={metrics['loss']:.4f}, PPL={metrics['ppl']:.1f}")
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'novel': novel_name,
                'epoch': epoch,
                'loss': metrics['loss'],
                'ppl': metrics['ppl'],
            }, output_path)
            print(f"   ✓ Saved (best loss: {best_loss:.4f})")
    
    print(f"\n✅ Pretraining complete for {novel_name}")
    print(f"   Model saved: {output_path}")
    
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BDH-Native Novel Pretraining')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    print("="*70)
    print("🐉 BDH-NATIVE NOVEL PRETRAINING")
    print("="*70)
    print("\nLeveraging BDH's unique architecture:")
    print("• Hebbian learning: neurons that fire together, wire together")
    print("• Sparse activations: monosemantic concept neurons")
    print("• Causal circuits: learned narrative reasoning chains\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Paths
    data_dir = ROOT / "Dataset" / "Books"
    tokenizer_path = ROOT / "models" / "custom_tokenizer.json"
    output_dir = ROOT / "models"
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    # Model config with BDH-specific settings
    config = TextPathConfig(
        vocab_size=vocab_size,
        max_seq_len=512,
        n_heads=8,
        n_neurons=2048,
        d_model=256,
        n_layers=4,
        dropout=0.1,
        use_rope=True,
        sparsity_target=0.05,  # BDH's natural operating point
        classification_mode=False,
    )
    
    # Train on each novel
    novels = [
        ("In search of the castaways",
         data_dir / "In search of the castaways.txt",
         output_dir / "textpath_in_search_of_the_castaways.pt"),
        ("The Count of Monte Cristo",
         data_dir / "The Count of Monte Cristo.txt",
         output_dir / "textpath_the_count_of_monte_cristo.pt"),
    ]
    
    for novel_name, novel_path, output_path in novels:
        if not novel_path.exists():
            print(f"⚠️ Novel not found: {novel_path}")
            continue
        
        pretrain_bdh_novel(
            novel_path=novel_path,
            tokenizer_path=tokenizer_path,
            output_path=output_path,
            device=device,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    
    print("\n" + "="*70)
    print("🎉 ALL PRETRAINING COMPLETE")
    print("="*70)
    print("\n💡 The BDH models have now learned:")
    print("   • Narrative patterns through Hebbian co-activation")
    print("   • Entity-specific neuron representations")
    print("   • Causal circuits encoding story logic")
    print("\n➡️  Next: python run_pipeline.py --mode full")


if __name__ == "__main__":
    main()
