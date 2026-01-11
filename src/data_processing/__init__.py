"""
Data processing module for RAG pipeline and dataset creation.
"""
from .retrieval import PathwayNovelRetriever
from .classification_dataset import ConsistencyDataset
from .build_retrievers import build_pathway_retrievers, build_retrievers_from_config
from .entity_threading import (
    create_character_threads,
    create_all_threads,
    extract_character_paragraphs
)

__all__ = [
    'PathwayNovelRetriever',
    'ConsistencyDataset', 
    'build_pathway_retrievers',
    'build_retrievers_from_config',
    'create_character_threads',
    'create_all_threads',
    'extract_character_paragraphs'
]

