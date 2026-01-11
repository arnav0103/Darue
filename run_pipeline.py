"""
Main execution script for the KDSH project.
Handles pretraining, calibration, evaluation, and visualization.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# Internal modules
from src.config import get_config, KDSHConfiguration
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

class ProjectRunner:
    """
    Controller class for the narrative consistency pipeline.
    """
    
    def __init__(self, config: KDSHConfiguration):
        self.config = config
        self.device = torch.device(config.device)
        self.retrievers = None
        
        print("\n" + "*" * 50)
        print(" KDSH System Initialized")
        print(f" Device: {self.device}")
        print("*" * 50 + "\n")

    def _ensure_retrievers(self):
        """Lazy load retrievers only when needed."""
        if self.retrievers is None:
            print("Initializing Retrieval System...")
            self.retrievers = build_pathway_retrievers(
                novels_dir=self.config.books_path,
                chunk_size=self.config.chunk_limit,
                overlap=self.config.chunk_overlap
            )

    def run_pretraining(self):
        """
        Execute the pretraining phase on novel text + entity threads.
        """
        print("\n>>> Starting Model Pretraining Phase")
        
        tokenizer = Tokenizer.from_file(str(self.config.tokenizer_file))
        vocab_size = tokenizer.get_vocab_size()
        
        threads_output = ROOT / "Dataset" / "entity_threads"
        
        # Configure the base model
        model_cfg = TextPathConfig(
            vocab_size=vocab_size,
            max_seq_len=self.config.sequence_length,
            n_heads=8,
            n_neurons=2048,
            d_model=256,
            n_layers=4,
            dropout=0.1,
            use_rope=True,
            sparsity_target=0.05,
            classification_mode=False
        )
        
        novels_to_process = [
            ("In search of the castaways",
             self.config.books_path / "In search of the castaways.txt",
             self.config.checkpoint_dir / "textpath_in_search_of_the_castaways.pt"),
            ("The Count of Monte Cristo",
             self.config.books_path / "The Count of Monte Cristo.txt",
             self.config.checkpoint_dir / "textpath_the_count_of_monte_cristo.pt"),
        ]
        
        for name, input_path, model_path in novels_to_process:
            if not input_path.exists():
                print(f"[!] {name} source file missing at {input_path}")
                continue
                
            print(f"\nProcessing: {name}")
            
            # Thread extraction
            print("  - Extracting character narratives...")
            novel_threads_dir = threads_output / input_path.stem
            thread_files = create_character_threads(
                novel_path=input_path,
                output_dir=novel_threads_dir,
                min_paragraphs=3
            )
            
            # Training
            print("  - Beginning BDH training...")
            pretrain_bdh_novel(
                novel_path=input_path,
                tokenizer_path=self.config.tokenizer_file,
                output_path=model_path,
                device=self.device,
                config=model_cfg,
                epochs=self.config.pretrain_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.lr,
                thread_paths=thread_files,
                thread_weight=2.0
            )

    def run_calibration(self):
        """Train the logistic regression calibrator."""
        self._ensure_retrievers()
        print("\n>>> Training Calibration Model")
        
        run_calibration_training(
            models_dir=str(self.config.checkpoint_dir),
            train_csv=str(self.config.train_data),
            novels_dir=str(self.config.books_path),
            tokenizer_path=str(self.config.tokenizer_file),
            retrievers=self.retrievers,
            device=self.config.device,
            output_path=str(self.config.calibration_model_path),
            top_k_retrieval=self.config.retrieval_k
        )

    def generate_predictions(self):
        """Run inference on the test set."""
        self._ensure_retrievers()
        print("\n>>> Generating Predictions")
        
        run_prediction(
            models_dir=str(self.config.checkpoint_dir),
            test_csv=str(self.config.test_data),
            tokenizer_path=str(self.config.tokenizer_file),
            retrievers=self.retrievers,
            device=self.config.device,
            output_path=str(self.config.prediction_output),
            top_k_retrieval=self.config.retrieval_k
        )

    def evaluate_model(self):
        """Run evaluation on validation split."""
        self._ensure_retrievers()
        print("\n>>> Evaluating Logic")
        
        run_evaluation(
            models_dir=str(self.config.checkpoint_dir),
            train_csv=str(self.config.train_data),
            tokenizer_path=str(self.config.tokenizer_file),
            retrievers=self.retrievers,
            device=self.config.device,
            top_k_retrieval=self.config.retrieval_k
        )

    def visualize_results(self):
        """Create analysis plots."""
        print("\n>>> Generating Visual Analysis")
        self._ensure_retrievers()
        
        from sklearn.metrics import accuracy_score, f1_score
        
        viz_root = ROOT / "visualizations"
        viz_root.mkdir(parents=True, exist_ok=True)
        
        tokenizer = Tokenizer.from_file(str(self.config.tokenizer_file))
        
        # Load one model to initialize scorer
        ckpt_files = list(self.config.checkpoint_dir.glob("textpath_*.pt"))
        if not ckpt_files:
            print("Error: No pretrained models found.")
            return

        print(f"Loading base model from {ckpt_files[0].name} for scoring...")
        checkpoint = torch.load(ckpt_files[0], map_location=self.device, weights_only=False)
        m_conf = checkpoint.get('config') or TextPathConfig(
            vocab_size=tokenizer.get_vocab_size(), 
            classification_mode=False
        )
        m_conf.classification_mode = False
        
        model = TextPath(m_conf)
        # Clean state dict
        state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                 if not k.startswith('classifier_head.')}
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        
        scorer = ConsistencyScorer(model=model, tokenizer=tokenizer, device=self.config.device)
        
        calib_model = load_calibration_model(str(self.config.calibration_model_path))
        
        # Validation set preparation
        df = pd.read_csv(self.config.train_data)
        val_idx = int(len(df) * 0.8)
        val_df = df.iloc[val_idx:].reset_index(drop=True)
        
        results = {
            'preds': [], 'labels': [], 'probs': [], 
            'deltas': [], 'features': [], 'novels': []
        }
        
        print(f"Processing {len(val_df)} validation samples...")
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
            txt = row['content']
            novel = row['book_name']
            
            chunks, scores = _retrieve_chunks(txt, novel, self.retrievers, self.config.retrieval_k)
            
            if chunks:
                feats = scorer.get_features(txt, chunks, scores)
                p, prob = predict_with_calibration(scorer, calib_model, txt, chunks, scores)
                delta = feats[0]
            else:
                feats = [0.0] * 4
                p, prob = 1, 0.5
                delta = 0.0
                
            results['preds'].append(p)
            results['labels'].append(1 if row['label'] == 'consistent' else 0)
            results['probs'].append(prob)
            results['deltas'].append(delta)
            results['features'].append(feats)
            results['novels'].append(novel)
            
        # Plotting
        print("Creating plots...")
        plot_delta_distribution(results['deltas'], results['labels'], str(viz_root / "delta_distribution.png"))
        plot_confusion_matrix(results['labels'], results['preds'], str(viz_root / "confusion_matrix.png"))
        plot_calibration_curve(results['labels'], results['probs'], str(viz_root / "calibration_curve.png"))
        
        if hasattr(calib_model.named_steps['classifier'], 'coef_'):
            plot_feature_importance(calib_model.named_steps['classifier'].coef_, str(viz_root / "feature_importance.png"))
            
        create_evaluation_dashboard(
            results['labels'], results['preds'], results['probs'], 
            results['deltas'], np.array(results['features']), 
            getattr(calib_model.named_steps['classifier'], 'coef_', None),
            results['novels'], str(viz_root)
        )
        
        acc = accuracy_score(results['labels'], results['preds'])
        print(f"Validation Accuracy: {acc:.2%}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='KDSH Execution Manager')
    parser.add_argument('--mode', default='full', 
                        choices=['pretrain', 'train', 'predict', 'evaluate', 'visualize', 'full'])
    
    # Overrides
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--pretrain-epochs', type=int, help='Pretraining epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = get_config()
    set_seed(config.seed)
    
    # Apply CLI overrides
    if args.pretrain_epochs: config.pretrain_epochs = args.pretrain_epochs
    if args.epochs: config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.lr: config.lr = args.lr
    
    runner = ProjectRunner(config)
    
    actions = {
        'pretrain': [runner.run_pretraining],
        'train': [runner.run_calibration],
        'predict': [runner.generate_predictions],
        'evaluate': [runner.evaluate_model],
        'visualize': [runner.visualize_results],
        'full': [
            runner.run_calibration,
            runner.evaluate_model,
            runner.visualize_results,
            runner.generate_predictions
        ]
    }
    
    # Run selected actions
    for action in actions[args.mode]:
        action()
        
    print("\n[+] Process Completed Successfully.")

if __name__ == '__main__':
    main()
