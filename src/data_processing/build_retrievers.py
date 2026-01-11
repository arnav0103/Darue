"""
Pathway RAG retriever building utilities.
Handles initialization of retrievers for all novels.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from typing import Dict
from .retrieval import PathwayNovelRetriever


def build_pathway_retrievers(
    novels_dir: Path,
    chunk_size: int = 200,
    overlap: int = 50
) -> Dict[str, PathwayNovelRetriever]:
    """
    Build Pathway-based retrievers for all novels in a directory.
    
    Args:
        novels_dir: Directory containing novel .txt files
        chunk_size: Size of chunks in words
        overlap: Overlap between chunks in words
        
    Returns:
        Dictionary mapping novel names to PathwayNovelRetriever instances
    """
    print("="*60)
    print("Building Pathway RAG Retrievers")
    print("="*60)
    
    retrievers = {}
    novels_dir = Path(novels_dir)
    
    for novel_file in novels_dir.glob('*.txt'):
        novel_name = novel_file.stem
        print(f"\nProcessing: {novel_name}")
        retrievers[novel_name] = PathwayNovelRetriever(
            novel_path=novel_file,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    print(f"\n✅ Built {len(retrievers)} Pathway retrievers")
    return retrievers


def build_retrievers_from_config(config) -> Dict[str, PathwayNovelRetriever]:
    """
    Build retrievers using a PipelineConfig object.
    
    Args:
        config: PipelineConfig instance or dict with novels_dir, chunk_size, overlap
        
    Returns:
        Dictionary of retrievers
    """
    if hasattr(config, 'novels_dir'):
        # PipelineConfig object
        return build_pathway_retrievers(
            novels_dir=config.novels_dir,
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
    else:
        # Dictionary
        return build_pathway_retrievers(
            novels_dir=config['novels_dir'],
            chunk_size=config['chunk_size'],
            overlap=config['overlap']
        )


__all__ = ['build_pathway_retrievers', 'build_retrievers_from_config']
