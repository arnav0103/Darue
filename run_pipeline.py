"""
End-to-End Pipeline for Narrative Consistency Detection
========================================================
Complete pipeline using Pathway framework for RAG and BDH for scoring.

NEW ARCHITECTURE (Generative Reasoning):
- Pretrain: BDH Language Model on (Raw Novel + Entity Threads)
- Score: Perplexity Delta - conditioning on backstory should reduce loss
- Train: Logistic Regression calibration on [Delta, Cosine_Similarity] features
- Predict: Delta -> Calibration Model -> 0/1

Usage:
    python run_pipeline.py --mode pretrain   # Pretrain BDH with entity threading
    python run_pipeline.py --mode train      # Train calibration model
    python run_pipeline.py --mode predict    # Generate predictions
    python run_pipeline.py --mode evaluate   # Evaluate on validation set
    python run_pipeline.py --mode visualize  # Generate evaluation visualizations
    python run_pipeline.py --mode full       # Train + Evaluate + Visualize + Predict
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from tokenizers import Tokenizer

# Import modular components
from src.config import PipelineConfig, get_config
from src.data_processing import build_pathway_retrievers, create_character_threads
from src.models.textpath import TextPath, TextPathConfig
from src.models.pretrain_bdh_native import pretrain_bdh_novel
from src.evaluation import run_evaluation, run_prediction
from src.training import (
    run_calibration_training,
    load_calibration_model,
    predict_with_calibration
)
from src.training.calibration import _retrieve_chunks
from src.analysis import ConsistencyScorer
from src.visualization import (
    plot_delta_distribution,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_feature_importance,
    create_evaluation_dashboard
)
from src.utils import set_seed



# ============================================================
# Pipeline Mode Runners
# ============================================================

def do_pretraining(config: PipelineConfig):
    """
    Run BDH-native pretraining on novels with Entity Threading.
    
    This creates character threads for each novel and trains the model
    on a mixture of raw novel text + entity threads.
    """
    print("\n" + "=" * 60)
    print("PRETRAINING WITH ENTITY THREADING")
    print("=" * 60)
    
    # Paths
    novels_dir = config.novels_dir
    tokenizer_path = config.tokenizer_path
    models_dir = config.models_dir
    
    # Create entity threads output directory
    threads_dir = ROOT / "Dataset" / "entity_threads"
    
    # Load tokenizer to get vocab size
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    # Model config
    model_config = TextPathConfig(
        vocab_size=vocab_size,
        max_seq_len=512,
        n_heads=8,
        n_neurons=2048,
        d_model=256,
        n_layers=4,
        dropout=0.1,
        use_rope=True,
        sparsity_target=0.05,
        classification_mode=False,  # LM mode for scoring
    )
    
    # Device
    device = torch.device(config.device)
    
    # Define novels to pretrain
    novels = [
        ("In search of the castaways",
         novels_dir / "In search of the castaways.txt",
         models_dir / "textpath_in_search_of_the_castaways.pt"),
        ("The Count of Monte Cristo",
         novels_dir / "The Count of Monte Cristo.txt",
         models_dir / "textpath_the_count_of_monte_cristo.pt"),
    ]
    
    for novel_name, novel_path, output_path in novels:
        if not novel_path.exists():
            print(f"Warning: Novel not found: {novel_path}")
            continue
        
        # Step 1: Create entity threads for this novel
        print(f"\nCreating entity threads for {novel_name}...")
        novel_threads_dir = threads_dir / novel_path.stem
        
        thread_paths = create_character_threads(
            novel_path=novel_path,
            output_dir=novel_threads_dir,
            min_paragraphs=3
        )
        
        # Step 2: Pretrain with mixed data (novel + threads)
        pretrain_bdh_novel(
            novel_path=novel_path,
            tokenizer_path=tokenizer_path,
            output_path=output_path,
            device=device,
            config=model_config,
            epochs=config.pretrain_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            thread_paths=thread_paths,
            thread_weight=2.0
        )
    
    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print("=" * 60)


def do_training(config: PipelineConfig, retrievers: dict):
    """
    Train calibration model using Perplexity Delta scoring.
    
    This replaces the old MLP training with lightweight Logistic Regression.
    """
    print("\n" + "=" * 60)
    print("TRAINING CALIBRATION MODEL")
    print("=" * 60)
    
    calibration_output = config.models_dir / "calibration_model.pkl"
    
    run_calibration_training(
        models_dir=str(config.models_dir),
        train_csv=str(config.train_csv),
        novels_dir=str(config.novels_dir),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        output_path=str(calibration_output),
        top_k_retrieval=config.top_k_retrieval
    )
    
    print(f"\nCalibration model saved to: {calibration_output}")


def do_prediction(config: PipelineConfig, retrievers: dict):
    """
    Generate predictions using Perplexity Delta + Calibration Model.
    """
    run_prediction(
        models_dir=str(config.models_dir),
        test_csv=str(config.test_csv),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        output_path=str(config.output_predictions),
        top_k_retrieval=config.top_k_retrieval
    )


def do_evaluation(config: PipelineConfig, retrievers: dict):
    """
    Evaluate on validation set using Perplexity Delta + Calibration.
    """
    run_evaluation(
        models_dir=str(config.models_dir),
        train_csv=str(config.train_csv),
        tokenizer_path=str(config.tokenizer_path),
        retrievers=retrievers,
        device=config.device,
        top_k_retrieval=config.top_k_retrieval
    )


def do_visualization(config: PipelineConfig, retrievers: dict):
    """
    Generate evaluation visualizations using Perplexity Delta data.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    # Create output directory
    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    
    # Load pretrained model
    model_paths = list(config.models_dir.glob("textpath_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No pretrained models found in {config.models_dir}")
    
    checkpoint = torch.load(model_paths[0], map_location=config.device, weights_only=False)
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
    model.to(config.device)
    model.eval()
    
    # Create scorer
    scorer = ConsistencyScorer(
        model=model,
        tokenizer=tokenizer,
        device=config.device
    )
    
    # Load calibration model
    calibration_path = config.models_dir / "calibration_model.pkl"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration model not found: {calibration_path}")
    
    calibration_model = load_calibration_model(str(calibration_path))
    
    # Load validation data
    train_df = pd.read_csv(config.train_csv)
    val_size = int(0.2 * len(train_df))
    val_df = train_df.tail(val_size).reset_index(drop=True)
    print(f"Generating visualizations for {len(val_df)} validation samples")
    
    # Collect data for visualization
    predictions = []
    labels = []
    probabilities = []
    deltas = []
    features = []
    novel_names = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Collecting data"):
        backstory = row['content']
        novel_name = row['book_name']
        true_label = 1 if row['label'] == 'consistent' else 0
        
        # Retrieve chunks
        chunks, scores = _retrieve_chunks(
            backstory, novel_name, retrievers, config.top_k_retrieval
        )
        
        # Get features
        if chunks:
            feature_vec = scorer.get_features(backstory, chunks, scores)
            pred, prob = predict_with_calibration(
                scorer, calibration_model, backstory, chunks, scores
            )
            delta = feature_vec[0]  # delta_mean
        else:
            feature_vec = [0.0, 0.0, 0.0, 0.0]
            pred, prob = 1, 0.5
            delta = 0.0
        
        predictions.append(pred)
        labels.append(true_label)
        probabilities.append(prob)
        deltas.append(delta)
        features.append(feature_vec)
        novel_names.append(novel_name)
    
    features = np.array(features)
    
    # Get calibration model coefficients
    coefficients = None
    try:
        coefficients = calibration_model.named_steps['classifier'].coef_
    except:
        pass
    
    # Generate individual plots
    print("\nGenerating plots...")
    
    # 1. Delta Distribution
    plot_delta_distribution(
        deltas=deltas,
        labels=labels,
        save_path=str(viz_dir / "delta_distribution.png")
    )
    
    # 2. Confusion Matrix
    plot_confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        save_path=str(viz_dir / "confusion_matrix.png")
    )
    
    # 3. Calibration Curve
    plot_calibration_curve(
        y_true=labels,
        y_prob=probabilities,
        save_path=str(viz_dir / "calibration_curve.png")
    )
    
    # 4. Feature Importance
    if coefficients is not None:
        plot_feature_importance(
            coefficients=coefficients,
            save_path=str(viz_dir / "feature_importance.png")
        )
    
    # 5. Comprehensive Dashboard
    create_evaluation_dashboard(
        y_true=labels,
        y_pred=predictions,
        y_prob=probabilities,
        deltas=deltas,
        features=features,
        coefficients=coefficients,
        novel_names=novel_names,
        output_dir=str(viz_dir)
    )
    
    print(f"\n All visualizations saved to: {viz_dir}")
    
    # Print summary metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Narrative Consistency Detection Pipeline (Generative Reasoning)'
    )
    parser.add_argument(
        '--mode', 
        choices=['pretrain', 'train', 'predict', 'evaluate', 'visualize', 'full'], 
        default='full',
        help='Pipeline mode: pretrain, train, predict, evaluate, visualize, or full'
    )
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=50,
        help='Number of epochs for BDH pretraining (default: 50)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs for classifier training (unused in new architecture)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (default: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: 1e-4)'
    )
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Override with command-line arguments
    if args.pretrain_epochs != 50:
        config.pretrain_epochs = args.pretrain_epochs
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # Print header
    print("=" * 60)
    print("Narrative Consistency Detection Pipeline")
    print("   Generative Reasoning with Perplexity Delta")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Mode: {args.mode}")
    
    # Pretraining mode
    if args.mode == 'pretrain':
        do_pretraining(config)
        return
    
    # Build retrievers for all other modes
    retrievers = build_pathway_retrievers(
        novels_dir=config.novels_dir,
        chunk_size=config.chunk_size,
        overlap=config.overlap
    )
    
    # Execute requested mode
    if args.mode == 'train':
        do_training(config, retrievers)
    elif args.mode == 'predict':
        do_prediction(config, retrievers)
    elif args.mode == 'evaluate':
        do_evaluation(config, retrievers)
    elif args.mode == 'visualize':
        do_visualization(config, retrievers)
    else:  # full
        do_training(config, retrievers)
        do_evaluation(config, retrievers)
        do_visualization(config, retrievers)
        do_prediction(config, retrievers)
    
    print("\n" + "=" * 60)
    print(" Pipeline Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
