"""
BDH-native pretraining module.
Provides interface to run pretraining on novels.
"""

import subprocess
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]


def run_pretraining(
    epochs: int = 50,
    batch_size: int = 4,
    pretrain_script: Optional[Path] = None,
    verbose: bool = True
) -> bool:
    """
    Run BDH-native pretraining on novels.
    
    Leverages BDH's unique properties:
    - Hebbian learning: neurons that fire together, wire together
    - Sparse activations (~5%): monosemantic concept neurons
    - Causal circuits (Gx = E @ Dx): learned narrative reasoning
    
    Args:
        epochs: Number of pretraining epochs
        batch_size: Batch size for pretraining
        pretrain_script: Path to pretraining script (default: src/models/pretrain_bdh_native.py)
        verbose: Whether to print progress
        
    Returns:
        True if pretraining succeeded, False otherwise
    """
    if verbose:
        print("\n" + "="*60)
        print("PHASE 0: BDH-Native Pretraining")
        print("="*60)
        print("\n Leveraging BDH's unique properties:")
        print("   • Hebbian learning: neurons that fire together, wire together")
        print("   • Sparse activations (~5%): monosemantic concept neurons")
        print("   • Causal circuits (Gx = E @ Dx): learned narrative reasoning")
    
    if pretrain_script is None:
        pretrain_script = ROOT / 'src' / 'models' / 'pretrain_bdh_native.py'
    
    cmd = [
        'python', str(pretrain_script),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
    ]
    
    if verbose:
        print(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(ROOT))
    
    if result.returncode != 0:
        if verbose:
            print("❌ Pretraining failed!")
        return False
    
    if verbose:
        print("\n✅ Pretraining complete!")
    return True


def run_pretraining_from_config(config, verbose: bool = True) -> bool:
    """
    Run pretraining using a PipelineConfig object.
    
    Args:
        config: PipelineConfig instance or dict
        verbose: Whether to print progress
        
    Returns:
        True if pretraining succeeded
    """
    if hasattr(config, 'pretrain_epochs'):
        epochs = config.pretrain_epochs
        batch_size = config.batch_size
    else:
        epochs = config.get('pretrain_epochs', 50)
        batch_size = config.get('batch_size', 4)
    
    return run_pretraining(
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )


__all__ = ['run_pretraining', 'run_pretraining_from_config']
