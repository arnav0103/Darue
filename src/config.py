"""
Configuration module for the KDSH pipeline.
Centralizes all configuration settings and paths.
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    Contains all paths and hyperparameters.
    """
    
    # ========================================
    # Paths
    # ========================================
    novels_dir: Path = field(default_factory=lambda: ROOT / 'Dataset' / 'Books')
    train_csv: Path = field(default_factory=lambda: ROOT / 'Dataset' / 'train.csv')
    test_csv: Path = field(default_factory=lambda: ROOT / 'Dataset' / 'test.csv')
    tokenizer_path: Path = field(default_factory=lambda: ROOT / 'models' / 'custom_tokenizer.json')
    models_dir: Path = field(default_factory=lambda: ROOT / 'models')
    output_model: Path = field(default_factory=lambda: ROOT / 'models' / 'calibration_model.pkl')
    output_predictions: Path = field(default_factory=lambda: ROOT / 'results.csv')
    
    # ========================================
    # Training Hyperparameters
    # ========================================
    batch_size: int = 4
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_tokens: int = 512  # Constrained by pretrained model's max_seq_len
    
    # Freezing strategy
    freeze_bdh: bool = True
    unfreeze_after_epoch: int = 5
    unfreeze_lr_multiplier: float = 0.1
    
    # Class weights for imbalanced data
    class_weight_inconsistent: float = 1.7
    class_weight_consistent: float = 1.0
    
    # ========================================
    # RAG (Pathway) Settings
    # ========================================
    chunk_size: int = 200   # ~200 words fits ~250 tokens
    overlap: int = 50
    top_k_retrieval: int = 2  # Retrieve top 2 most relevant passages
    
    # ========================================
    # Pretraining
    # ========================================
    pretrain_epochs: int = 50
    
    # ========================================
    # Device (auto-detected)
    # ========================================
    device: str = field(default_factory=lambda: (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    ))
    
    # Random seed
    seed: int = 42
    
    @property
    def class_weights(self):
        """Get class weights as tuple."""
        return (self.class_weight_inconsistent, self.class_weight_consistent)
    
    def to_dict(self):
        """Convert to dictionary for backwards compatibility."""
        return {
            'novels_dir': self.novels_dir,
            'train_csv': self.train_csv,
            'test_csv': self.test_csv,
            'tokenizer_path': self.tokenizer_path,
            'models_dir': self.models_dir,
            'output_model': self.output_model,
            'output_predictions': self.output_predictions,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'max_tokens': self.max_tokens,
            'freeze_bdh': self.freeze_bdh,
            'unfreeze_after_epoch': self.unfreeze_after_epoch,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'top_k_retrieval': self.top_k_retrieval,
            'pretrain_epochs': self.pretrain_epochs,
            'device': self.device,
        }


def get_config() -> PipelineConfig:
    """
    Get the default pipeline configuration.
    
    Returns:
        PipelineConfig instance with all defaults
    """
    return PipelineConfig()


def get_config_dict() -> dict:
    """
    Get configuration as dictionary for backwards compatibility.
    
    Returns:
        Dictionary with configuration values
    """
    return get_config().to_dict()


__all__ = ['PipelineConfig', 'get_config', 'get_config_dict', 'ROOT']
