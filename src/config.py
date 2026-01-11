"""
Global configuration settings for the project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class KDSHConfiguration:
    """
    Main configuration class containing all parameters and paths.
    """
    
    # Paths
    dataset_path: Path = field(default_factory=lambda: BASE_DIR / 'Dataset')
    books_path: Path = field(default_factory=lambda: BASE_DIR / 'Dataset' / 'Books')
    train_data: Path = field(default_factory=lambda: BASE_DIR / 'Dataset' / 'train.csv')
    test_data: Path = field(default_factory=lambda: BASE_DIR / 'Dataset' / 'test.csv')
    
    tokenizer_file: Path = field(default_factory=lambda: BASE_DIR / 'models' / 'custom_tokenizer.json')
    checkpoint_dir: Path = field(default_factory=lambda: BASE_DIR / 'models')
    calibration_model_path: Path = field(default_factory=lambda: BASE_DIR / 'models' / 'calibration_model.pkl')
    prediction_output: Path = field(default_factory=lambda: BASE_DIR / 'results.csv')
    
    # Model Parameters
    batch_size: int = 4
    num_epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.01
    sequence_length: int = 512
    
    # RAG settings
    chunk_limit: int = 200
    chunk_overlap: int = 50
    retrieval_k: int = 2
    
    # Pretraining
    pretrain_epochs: int = 50
    
    # System
    device: str = field(default_factory=lambda: (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    ))
    seed: int = 42

    # Compat getters for old code that might expect specific names
    @property
    def novels_dir(self): return self.books_path
    @property
    def train_csv(self): return self.train_data
    @property
    def test_csv(self): return self.test_data
    @property
    def tokenizer_path(self): return self.tokenizer_file
    @property
    def models_dir(self): return self.checkpoint_dir
    @property
    def output_predictions(self): return self.prediction_output
    @property
    def learning_rate(self): return self.lr
    @property
    def max_tokens(self): return self.sequence_length
    @property
    def chunk_size(self): return self.chunk_limit
    @property
    def overlap(self): return self.chunk_overlap
    @property
    def top_k_retrieval(self): return self.retrieval_k
    @property
    def epochs(self): return self.num_epochs

# Alias for compatibility if needed, or simply use the new name
PipelineConfig = KDSHConfiguration

def get_config():
    return KDSHConfiguration()

