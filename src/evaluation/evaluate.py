"""
Evaluation Module for Generative Reasoning
===========================================
Modular evaluation utilities for the Perplexity Delta scoring approach.

Provides:
- run_evaluation: Evaluate model on validation set
- run_prediction: Generate predictions on test set
- compute_metrics: Calculate accuracy, F1, precision, recall
- save_predictions: Save predictions to CSV
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

import torch
from tokenizers import Tokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.textpath import TextPath, TextPathConfig
from src.analysis.consistency_scorer import ConsistencyScorer
from src.training.calibration import (
    load_calibration_model,
    predict_with_calibration,
    _retrieve_chunks
)


# ============================================================
# Metrics Computation
# ============================================================

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes (default: ['Contradictory', 'Consistent'])
        
    Returns:
        Dictionary with metrics: accuracy, f1, precision, recall, 
        confusion_matrix, classification_report
    """
    if class_names is None:
        class_names = ['Contradictory', 'Consistent']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        ),
        'classification_report_str': classification_report(
            y_true, y_pred,
            target_names=class_names
        )
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str = "EVALUATION RESULTS"):
    """
    Print formatted evaluation metrics.
    
    Args:
        metrics: Dictionary from compute_metrics()
        title: Title for the report
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"            Contra  Consist")
    print(f"Actual Contra  {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"       Consist {cm[1, 0]:4d}    {cm[1, 1]:4d}")
    
    print("\nClassification Report:")
    print(metrics['classification_report_str'])


# ============================================================
# Model Loading Utilities
# ============================================================

def load_scorer_and_calibration(
    models_dir: Path,
    tokenizer_path: Path,
    device: str
) -> Tuple[ConsistencyScorer, object]:
    """
    Load the ConsistencyScorer and calibration model.
    
    Args:
        models_dir: Directory containing pretrained models
        tokenizer_path: Path to tokenizer JSON
        device: Device to load models on
        
    Returns:
        Tuple of (ConsistencyScorer, calibration_model)
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load pretrained BDH model
    model_paths = list(models_dir.glob("textpath_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No pretrained models found in {models_dir}")
    
    checkpoint = torch.load(model_paths[0], map_location=device, weights_only=False)
    model_config = checkpoint.get('config')
    
    if model_config is None:
        model_config = TextPathConfig(
            vocab_size=tokenizer.get_vocab_size(),
            classification_mode=False
        )
    else:
        model_config.classification_mode = False
    
    model = TextPath(model_config)
    
    # Load state dict, filtering out classifier head weights if not in classification mode
    checkpoint_state = checkpoint['model_state_dict']
    if not model_config.classification_mode:
        # Remove classifier head weights from checkpoint if they exist
        checkpoint_state = {
            k: v for k, v in checkpoint_state.items()
            if not k.startswith('classifier_head.')
        }
    
    model.load_state_dict(checkpoint_state, strict=False)
    model.to(device)
    model.eval()
    
    # Create scorer
    scorer = ConsistencyScorer(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Load calibration model
    calibration_path = models_dir / "calibration_model.pkl"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration model not found: {calibration_path}")
    
    calibration_model = load_calibration_model(str(calibration_path))
    
    return scorer, calibration_model


# ============================================================
# Evaluation Functions
# ============================================================

def run_evaluation(
    models_dir: str,
    train_csv: str,
    tokenizer_path: str,
    retrievers: dict,
    device: str,
    top_k_retrieval: int = 2,
    val_ratio: float = 0.2,
    verbose: bool = True
) -> Dict:
    """
    Run evaluation on validation set using Perplexity Delta scoring.
    
    Args:
        models_dir: Directory containing pretrained models
        train_csv: Path to training CSV
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to run on
        top_k_retrieval: Number of chunks to retrieve
        val_ratio: Proportion of data to use for validation
        verbose: Whether to print progress and results
        
    Returns:
        Dictionary with metrics and prediction details
    """
    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
    
    models_dir = Path(models_dir)
    
    # Load scorer and calibration model
    scorer, calibration_model = load_scorer_and_calibration(
        models_dir=models_dir,
        tokenizer_path=Path(tokenizer_path),
        device=device
    )
    
    # Load validation data
    train_df = pd.read_csv(train_csv)
    val_size = int(val_ratio * len(train_df))
    val_df = train_df.tail(val_size).reset_index(drop=True)
    
    if verbose:
        print(f"Evaluating on {len(val_df)} validation samples")
    
    # Run evaluation
    predictions = []
    labels = []
    probabilities = []
    deltas = []
    features = []
    
    iterator = tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating") if verbose else val_df.iterrows()
    
    for idx, row in iterator:
        backstory = row['content']
        novel_name = row['book_name']
        true_label = 1 if row['label'] == 'consistent' else 0
        
        # Retrieve chunks
        chunks, scores = _retrieve_chunks(
            backstory, novel_name, retrievers, top_k_retrieval
        )
        
        # Get features and prediction
        if chunks:
            feature_vec = scorer.get_features(backstory, chunks, scores)
            pred, prob = predict_with_calibration(
                scorer, calibration_model, backstory, chunks, scores
            )
        else:
            feature_vec = [0.0, 0.0, 0.0, 0.0]
            pred, prob = 1, 0.5
        
        predictions.append(pred)
        labels.append(true_label)
        probabilities.append(prob)
        deltas.append(feature_vec[0])  # delta_mean
        features.append(feature_vec)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions)
    
    if verbose:
        print_metrics(metrics)
    
    # Add extra details
    metrics['predictions'] = predictions
    metrics['labels'] = labels
    metrics['probabilities'] = probabilities
    metrics['deltas'] = deltas
    metrics['features'] = np.array(features)
    
    return metrics


def run_prediction(
    models_dir: str,
    test_csv: str,
    tokenizer_path: str,
    retrievers: dict,
    device: str,
    output_path: str,
    top_k_retrieval: int = 2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate predictions on test set.
    
    Args:
        models_dir: Directory containing pretrained models
        test_csv: Path to test CSV
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to run on
        output_path: Path to save predictions CSV
        top_k_retrieval: Number of chunks to retrieve
        verbose: Whether to print progress
        
    Returns:
        DataFrame with predictions
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)
    
    models_dir = Path(models_dir)
    
    # Load scorer and calibration model
    scorer, calibration_model = load_scorer_and_calibration(
        models_dir=models_dir,
        tokenizer_path=Path(tokenizer_path),
        device=device
    )
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    
    if verbose:
        print(f"Predicting on {len(test_df)} test samples")
    
    # Generate predictions
    predictions = []
    probabilities = []
    
    iterator = tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting") if verbose else test_df.iterrows()
    
    for idx, row in iterator:
        backstory = row['content']
        novel_name = row['book_name']
        
        # Retrieve chunks
        chunks, scores = _retrieve_chunks(
            backstory, novel_name, retrievers, top_k_retrieval
        )
        
        # Predict
        if chunks:
            pred, prob = predict_with_calibration(
                scorer, calibration_model, backstory, chunks, scores
            )
        else:
            pred, prob = 1, 0.5
        
        predictions.append(pred)
        probabilities.append(prob)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Story ID': test_df['id'],
        'Prediction': predictions
    })
    results = results.sort_values('Story ID').reset_index(drop=True)
    
    # Save predictions
    output_path = Path(output_path)
    results.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\nPredictions saved to: {output_path}")
        print("\nPrediction distribution:")
        print(results['Prediction'].value_counts())
    
    return results


def save_predictions(
    ids: List[int],
    predictions: List[int],
    output_path: str,
    label_map: Dict[int, str] = None
) -> pd.DataFrame:
    """
    Save predictions to CSV file.
    
    Args:
        ids: Sample IDs
        predictions: Predicted labels (0 or 1)
        output_path: Path to save CSV
        label_map: Optional mapping from int to string labels
        
    Returns:
        DataFrame with predictions
    """
    if label_map is None:
        # User requested format: 1=consistent, 0=inconsistent
        pass 
    
    results = pd.DataFrame({
        'Story ID': ids,
        'Prediction': predictions 
    })
    results = results.sort_values('id').reset_index(drop=True)
    results.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved to {output_path}")
    print("\nPrediction distribution:")
    print(results['Prediction'].value_counts())
    
    return results


# ============================================================
# Exports
# ============================================================

__all__ = [
    'compute_metrics',
    'print_metrics',
    'load_scorer_and_calibration',
    'run_evaluation',
    'run_prediction',
    'save_predictions'
]
