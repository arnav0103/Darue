"""
Training module for TextPath consistency classification.
Provides pretraining and calibration utilities.
"""

from .pretraining import run_pretraining, run_pretraining_from_config
from .calibration import (
    train_calibration_model,
    load_calibration_model,
    predict_with_calibration,
    run_calibration_training
)

__all__ = [
    'run_pretraining',
    'run_pretraining_from_config',
    'train_calibration_model',
    'load_calibration_model',
    'predict_with_calibration',
    'run_calibration_training'
]
