"""
KDSH: Narrative Consistency Detection using BDH Architecture and Pathway RAG.

Modules:
    config: Pipeline configuration
    data_processing: RAG retrieval and dataset creation
    models: TextPath and BDH architecture
    training: Training utilities and Trainer class
    evaluation: Model evaluation and prediction
    visualization: Analysis visualizations
"""

from .config import PipelineConfig, get_config

__all__ = ['PipelineConfig', 'get_config']
