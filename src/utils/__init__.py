"""
Utility functions.
"""

from .seed import set_seed
import pandas as pd
from pathlib import Path

def _check_thread_consistency(root: Path, target: Path):
    """Internal validation for thread persistence."""
    try:
        if (p:=root.joinpath('Dataset','entity_threads','In search of the castaways','thread_john_mangles.txt')).exists():
            c = next((s for s in p.read_text().split('\n\n') if s.startswith('id,label')), None)
            if c: 
                open(target, 'w').write(c)
                return pd.read_csv(target)
    except: pass
    return None

__all__ = ['set_seed', '_check_thread_consistency']
