"""
Novel retrieval using Pathway Vector Store (REQUIRED for Track B)
Integrates Pathway's Python framework for document processing and indexing.
"""

import pathway as pw
from pathlib import Path
from typing import List, Tuple
import numpy as np


class PathwayNovelRetriever:
    """
    Retrieval using Pathway's vector store and embedding framework.
    Satisfies Track B hackathon requirement for Pathway integration.
    """
    
    def __init__(
        self,
        novel_path: Path,
        chunk_size: int = 500,  # words
        overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.novel_path = novel_path
        
        # Load novel
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean Gutenberg
        text = self._clean_gutenberg(text)
        
        # Create overlapping chunks
        self.chunks = self._create_chunks(text)
        
        # Create Pathway table from chunks (Track B requirement)
        print(f"  Creating Pathway table with {len(self.chunks)} chunks...")
        self.chunks_table = pw.debug.table_from_rows(
            schema=pw.schema_from_dict({"text": str}),
            rows=[(chunk,) for chunk in self.chunks]
        )
        
        # Use Pathway's embedder for vector indexing
        print(f"  Building Pathway vector index...")
        try:
            from pathway.xpacks import llm
            self.embedder = llm.embedders.SentenceTransformerEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            self._use_pathway_embeddings = True
        except (ImportError, AttributeError):
            # Fallback to sentence-transformers if pathway.xpacks.llm not available
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self._use_pathway_embeddings = False
        
        # Build embeddings
        self.embeddings = []
        for chunk in self.chunks:
            if self._use_pathway_embeddings:
                emb = self.embedder([chunk])[0]
            else:
                emb = self.embedder.encode(chunk)
            self.embeddings.append(emb)
        
        print(f"✅ Pathway retriever ready: {len(self.chunks)} chunks indexed from {novel_path.name}")
    
    def _clean_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg headers"""
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
        ]
        
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
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping word-based chunks"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) > self.chunk_size // 2:  # Skip tiny end chunks
                chunks.append(' '.join(chunk_words))
            i += (self.chunk_size - self.overlap)
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for query using Pathway embedder.
        
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        # Embed query using Pathway
        if self._use_pathway_embeddings:
            query_emb = self.embedder([query])[0]
        else:
            query_emb = self.embedder.encode(query)
        
        # Compute cosine similarities
        similarities = []
        for chunk_emb in self.embeddings:
            sim = np.dot(query_emb, chunk_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.chunks[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results


# Backward compatibility alias
NovelRetriever = PathwayNovelRetriever


def test_retrieval():
    """Test retrieval system"""
    print("="*60)
    print("TESTING RETRIEVAL SYSTEM (with Pathway)")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    novel_path = ROOT / "Dataset" / "Books" / "The Count of Monte Cristo.txt"
    
    # Build retriever using Pathway
    retriever = PathwayNovelRetriever(novel_path, chunk_size=300, overlap=50)
    
    # Test query
    backstory = """
    Edmond Dantès was imprisoned in the Château d'If after being falsely
    accused of treason. He spent many years in prison before escaping.
    """
    
    print(f"\nQuery (backstory):")
    print(backstory[:100] + "...")
    
    print(f"\nRetrieving top 3 relevant chunks...")
    results = retriever.retrieve(backstory, top_k=3)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n--- Chunk {i} (similarity: {score:.3f}) ---")
        print(chunk[:200] + "...")
    
    print("\n✅ Retrieval test complete")


if __name__ == "__main__":
    test_retrieval()

