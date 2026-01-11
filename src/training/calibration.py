"""
Calibration Module for Generative Reasoning
============================================
Trains a lightweight Logistic Regression model on top of Perplexity Delta features.

This replaces the deep learning classifier training loop with a simple
scikit-learn pipeline that:
1. Computes Delta + Cosine Similarity features for each training sample
2. Trains a Logistic Regression model on these features
3. Saves the calibration model for inference

The calibration model converts raw delta scores into probability estimates,
making the predictions more reliable and interpretable.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

import torch
from tokenizers import Tokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.textpath import TextPath, TextPathConfig
from src.analysis.consistency_scorer import ConsistencyScorer


def train_calibration_model(
    model: TextPath,
    tokenizer: Tokenizer,
    train_csv: str,
    retrievers: dict,
    device: str,
    output_path: str,
    top_k_retrieval: int = 2,
    verbose: bool = True
) -> Pipeline:
    """
    Train calibration model on training data.
    
    This function:
    1. Creates a ConsistencyScorer from the pretrained model
    2. Loops through train.csv to compute features for each sample
    3. Trains a Logistic Regression on [delta, cosine_similarity] features
    4. Saves the trained pipeline to disk
    
    Args:
        model: Pretrained TextPath model
        tokenizer: Tokenizer for the model
        train_csv: Path to training CSV with columns [id, book_name, content, label]
        retrievers: Dictionary of PathwayNovelRetriever by novel name
        device: Device to run inference on
        output_path: Path to save calibration model (.pkl)
        top_k_retrieval: Number of chunks to retrieve per sample
        verbose: Whether to print progress
        
    Returns:
        Trained sklearn Pipeline (Scaler + LogisticRegression)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING CALIBRATION MODEL")
        print("=" * 60)
    
    # Create scorer
    scorer = ConsistencyScorer(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Load training data
    df = pd.read_csv(train_csv)
    if verbose:
        print(f"Loaded {len(df)} training samples")
    
    # Collect features and labels
    features = []
    labels = []
    
    iterator = tqdm(df.iterrows(), total=len(df), desc="Computing features") if verbose else df.iterrows()
    
    for idx, row in iterator:
        backstory = row['content']
        novel_name = row['book_name']
        label = 1 if row['label'] == 'consistent' else 0
        
        # Retrieve relevant chunks
        chunks, retrieval_scores = _retrieve_chunks(
            backstory, novel_name, retrievers, top_k_retrieval
        )
        
        # Get feature vector
        if chunks:
            feature_vec = scorer.get_features(backstory, chunks, retrieval_scores)
        else:
            feature_vec = [0.0, 0.0, 0.0, 0.0]
        
        features.append(feature_vec)
        labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    if verbose:
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Cross-validation score
    if verbose:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full data
    pipeline.fit(X, y)
    
    # Feature importance (coefficients)
    if verbose:
        coef = pipeline.named_steps['classifier'].coef_[0]
        feature_names = ['delta_mean', 'delta_max', 'cosine_mean', 'retrieval_score']
        print("\nFeature coefficients:")
        for name, c in zip(feature_names, coef):
            print(f"  {name}: {c:.4f}")
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    
    if verbose:
        print(f"\nCalibration model saved to: {output_path}")
    
    return pipeline


def load_calibration_model(model_path: str) -> Pipeline:
    """
    Load a trained calibration model from disk.
    
    Args:
        model_path: Path to the saved .pkl file
        
    Returns:
        Trained sklearn Pipeline
    """
    return joblib.load(model_path)


def predict_with_calibration(
    scorer: ConsistencyScorer,
    calibration_model: Pipeline,
    backstory: str,
    novel_chunks: List[str],
    retrieval_scores: Optional[List[float]] = None
) -> Tuple[int, float]:
    """
    Predict consistency using scorer + calibration model.
    
    Args:
        scorer: ConsistencyScorer instance
        calibration_model: Trained calibration Pipeline
        backstory: The backstory/claim to check
        novel_chunks: Retrieved chunks from the novel
        retrieval_scores: Optional retrieval similarity scores
        
    Returns:
        Tuple of (prediction, probability):
        - prediction: 0 (contradict) or 1 (consistent)
        - probability: Probability of being consistent
    """
    # Get features
    features = scorer.get_features(backstory, novel_chunks, retrieval_scores)
    X = np.array([features])
    
    # Predict
    prediction = calibration_model.predict(X)[0]
    probability = calibration_model.predict_proba(X)[0, 1]  # P(consistent)
    
    return int(prediction), float(probability)


def _retrieve_chunks(
    query: str,
    novel_name: str,
    retrievers: dict,
    top_k: int
) -> Tuple[List[str], List[float]]:
    """
    Retrieve relevant chunks from the novel.
    
    Args:
        query: Query text (backstory)
        novel_name: Name of the novel
        retrievers: Dictionary of retriever instances
        top_k: Number of chunks to retrieve
        
    Returns:
        Tuple of (chunks, scores)
    """
    # Find matching retriever
    retriever = None
    
    # Try exact match first
    if novel_name in retrievers:
        retriever = retrievers[novel_name]
    else:
        # Try case-insensitive matching
        novel_name_lower = novel_name.lower()
        for key in retrievers.keys():
            if key.lower() == novel_name_lower:
                retriever = retrievers[key]
                break
            elif 'monte cristo' in novel_name_lower and 'monte cristo' in key.lower():
                retriever = retrievers[key]
                break
            elif 'castaways' in novel_name_lower and 'castaways' in key.lower():
                retriever = retrievers[key]
                break
    
    if retriever is None:
        # Fallback to first available
        if retrievers:
            retriever = list(retrievers.values())[0]
        else:
            return [], []
    
    # Retrieve chunks
    results = retriever.retrieve(query, top_k=top_k)
    
    if results:
        chunks = [chunk for chunk, score in results]
        scores = [float(score) for chunk, score in results]
        return chunks, scores
    
    return [], []


def run_calibration_training(
    models_dir: str,
    train_csv: str,
    novels_dir: str,
    tokenizer_path: str,
    retrievers: dict,
    device: str,
    output_path: str,
    top_k_retrieval: int = 2
) -> Pipeline:
    """
    High-level function to run calibration training.
    
    This loads the pretrained BDH model and trains a calibration model on top.
    
    Args:
        models_dir: Directory containing pretrained BDH models
        train_csv: Path to training CSV
        novels_dir: Directory with novel text files (unused, for compatibility)
        tokenizer_path: Path to tokenizer JSON
        retrievers: Dictionary of PathwayNovelRetriever instances
        device: Device to run on
        output_path: Path to save calibration model
        top_k_retrieval: Number of chunks to retrieve
        
    Returns:
        Trained calibration Pipeline
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Determine which model to load based on available pretrained models
    models_dir = Path(models_dir)
    
    # Try to find a pretrained model
    pretrained_paths = list(models_dir.glob("textpath_*.pt"))
    
    if not pretrained_paths:
        raise FileNotFoundError(f"No pretrained models found in {models_dir}")
    
    # Load first available model (we'll use same model for all novels for now)
    model_path = pretrained_paths[0]
    print(f"Loading pretrained model: {model_path.name}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    if config is None:
        config = TextPathConfig(
            vocab_size=vocab_size,
            classification_mode=False
        )
    else:
        config.classification_mode = False
    
    # Initialize and load model
    model = TextPath(config)
    
    # Load state dict, filtering out classifier head weights if not in classification mode
    checkpoint_state = checkpoint['model_state_dict']
    if not config.classification_mode:
        # Remove classifier head weights from checkpoint if they exist
        checkpoint_state = {
            k: v for k, v in checkpoint_state.items()
            if not k.startswith('classifier_head.')
        }
    
    model.load_state_dict(checkpoint_state, strict=False)
    model.to(device)
    model.eval()
    
    # Train calibration model
    return train_calibration_model(
        model=model,
        tokenizer=tokenizer,
        train_csv=train_csv,
        retrievers=retrievers,
        device=device,
        output_path=output_path,
        top_k_retrieval=top_k_retrieval
    )


# Export for module
__all__ = [
    'train_calibration_model',
    'load_calibration_model',
    'predict_with_calibration',
    'run_calibration_training'
]
