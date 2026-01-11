"""
Visualization Utilities for Generative Reasoning
=================================================
Modular visualization tools for analyzing the Perplexity Delta scoring approach.

Provides:
- Delta distribution plots
- Calibration curves
- Feature importance analysis
- Per-novel score comparisons
- Confusion matrix visualization
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================
# Color Schemes
# ============================================================

COLORS = {
    'consistent': '#2ecc71',      # Green
    'contradict': '#e74c3c',      # Red
    'neutral': '#3498db',         # Blue
    'background': '#f8f9fa',      # Light gray
    'grid': '#dee2e6',            # Gray
    'text': '#2c3e50',            # Dark blue-gray
}


# ============================================================
# Delta Distribution Visualization
# ============================================================

def plot_delta_distribution(
    deltas: List[float],
    labels: List[int],
    title: str = "Perplexity Delta Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the distribution of perplexity deltas for consistent vs contradictory samples.
    
    Args:
        deltas: List of delta scores
        labels: List of labels (0=contradict, 1=consistent)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Separate by label
    deltas = np.array(deltas)
    labels = np.array(labels)
    
    consistent_deltas = deltas[labels == 1]
    contradict_deltas = deltas[labels == 0]
    
    # Plot histograms
    bins = np.linspace(min(deltas), max(deltas), 30)
    
    ax.hist(consistent_deltas, bins=bins, alpha=0.7, 
            color=COLORS['consistent'], label=f'Consistent (n={len(consistent_deltas)})',
            edgecolor='white', linewidth=0.5)
    ax.hist(contradict_deltas, bins=bins, alpha=0.7,
            color=COLORS['contradict'], label=f'Contradictory (n={len(contradict_deltas)})',
            edgecolor='white', linewidth=0.5)
    
    # Add vertical line at zero
    ax.axvline(x=0, color=COLORS['text'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.02, ax.get_ylim()[1] * 0.95, 'Δ = 0', fontsize=10, color=COLORS['text'])
    
    # Styling
    ax.set_xlabel('Perplexity Delta (Δ)', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Frequency', fontsize=12, color=COLORS['text'])
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Add annotation
    ax.annotate('← Contradictory', xy=(min(deltas) * 0.7, ax.get_ylim()[1] * 0.8),
                fontsize=10, color=COLORS['contradict'], alpha=0.8)
    ax.annotate('Consistent →', xy=(max(deltas) * 0.5, ax.get_ylim()[1] * 0.8),
                fontsize=10, color=COLORS['consistent'], alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Feature Scatter Plot
# ============================================================

def plot_feature_scatter(
    features: np.ndarray,
    labels: List[int],
    feature_names: List[str] = None,
    title: str = "Feature Space Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot 2D scatter of features colored by label.
    
    Args:
        features: Feature matrix [n_samples, n_features]
        labels: List of labels
        feature_names: Names of features
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if feature_names is None:
        feature_names = ['delta_mean', 'delta_max', 'cosine_mean', 'retrieval_score']
    
    labels = np.array(labels)
    n_features = min(features.shape[1], 4)
    
    fig, axes = plt.subplots(1, n_features - 1, figsize=figsize, facecolor=COLORS['background'])
    if n_features == 2:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.set_facecolor(COLORS['background'])
        
        # Plot points
        consistent_mask = labels == 1
        contradict_mask = labels == 0
        
        ax.scatter(features[contradict_mask, 0], features[contradict_mask, i + 1],
                   c=COLORS['contradict'], alpha=0.6, s=30, label='Contradictory')
        ax.scatter(features[consistent_mask, 0], features[consistent_mask, i + 1],
                   c=COLORS['consistent'], alpha=0.6, s=30, label='Consistent')
        
        ax.set_xlabel(feature_names[0], fontsize=10, color=COLORS['text'])
        ax.set_ylabel(feature_names[i + 1], fontsize=10, color=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        if i == 0:
            ax.legend(loc='best', framealpha=0.9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Calibration Curve
# ============================================================

def plot_calibration_curve(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot calibration curve showing predicted probability vs actual frequency.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate actual frequency per bin
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Avoid division by zero
    bin_counts = np.maximum(bin_counts, 1)
    bin_freqs = bin_sums / bin_counts
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    
    # Plot actual calibration
    ax.plot(bin_centers, bin_freqs, 'o-', color=COLORS['neutral'], 
            linewidth=2, markersize=8, label='Model Calibration')
    
    # Add confidence region
    ax.fill_between(bin_centers, bin_freqs - 0.1, bin_freqs + 0.1, 
                    alpha=0.2, color=COLORS['neutral'])
    
    # Styling
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Fraction of Positives', fontsize=12, color=COLORS['text'])
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with counts and percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if class_names is None:
        class_names = ['Contradictory', 'Consistent']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#ffffff', COLORS['neutral']], N=256
    )
    
    # Plot heatmap
    im = ax.imshow(cm_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontsize=10, color=COLORS['text'])
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = 'white' if cm_norm[i, j] > 0.5 else COLORS['text']
            ax.text(j, i, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=12, color=text_color)
    
    # Styling
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Actual', fontsize=12, color=COLORS['text'])
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Feature Importance
# ============================================================

def plot_feature_importance(
    coefficients: np.ndarray,
    feature_names: List[str] = None,
    title: str = "Feature Importance (Calibration Model)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot feature importance from calibration model coefficients.
    
    Args:
        coefficients: Model coefficients
        feature_names: Names of features
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if feature_names is None:
        feature_names = ['delta_mean', 'delta_max', 'cosine_mean', 'retrieval_score']
    
    coefficients = np.array(coefficients).flatten()
    
    # Sort by absolute importance
    indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coef = coefficients[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Color based on sign
    colors = [COLORS['consistent'] if c > 0 else COLORS['contradict'] for c in sorted_coef]
    
    # Plot bars
    bars = ax.barh(range(len(sorted_coef)), sorted_coef, color=colors, edgecolor='white')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_coef)):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=10, color=COLORS['text'])
    
    # Styling
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Coefficient Value', fontsize=12, color=COLORS['text'])
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.axvline(x=0, color=COLORS['text'], linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x', color=COLORS['grid'])
    
    # Legend
    positive_patch = mpatches.Patch(color=COLORS['consistent'], label='→ Consistent')
    negative_patch = mpatches.Patch(color=COLORS['contradict'], label='→ Contradictory')
    ax.legend(handles=[positive_patch, negative_patch], loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Per-Novel Analysis
# ============================================================

def plot_per_novel_scores(
    novel_names: List[str],
    deltas: List[float],
    labels: List[int],
    title: str = "Score Distribution by Novel",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot score distributions grouped by novel.
    
    Args:
        novel_names: List of novel names for each sample
        deltas: List of delta scores
        labels: List of labels
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Group by novel
    novels = {}
    for name, delta, label in zip(novel_names, deltas, labels):
        # Normalize novel name
        if 'monte cristo' in name.lower():
            key = 'Monte Cristo'
        elif 'castaways' in name.lower():
            key = 'Castaways'
        else:
            key = name
        
        if key not in novels:
            novels[key] = {'consistent': [], 'contradict': []}
        
        if label == 1:
            novels[key]['consistent'].append(delta)
        else:
            novels[key]['contradict'].append(delta)
    
    # Create figure
    fig, axes = plt.subplots(1, len(novels), figsize=figsize, facecolor=COLORS['background'])
    if len(novels) == 1:
        axes = [axes]
    
    for ax, (novel_name, data) in zip(axes, novels.items()):
        ax.set_facecolor(COLORS['background'])
        
        # Box plot
        box_data = [data['contradict'], data['consistent']]
        bp = ax.boxplot(box_data, labels=['Contradict', 'Consistent'], patch_artist=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor(COLORS['contradict'])
        bp['boxes'][1].set_facecolor(COLORS['consistent'])
        
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax.set_title(novel_name, fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_ylabel('Delta Score', fontsize=10, color=COLORS['text'])
        ax.axhline(y=0, color=COLORS['text'], linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# Summary Dashboard
# ============================================================

def create_evaluation_dashboard(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
    deltas: List[float],
    features: np.ndarray,
    coefficients: np.ndarray = None,
    novel_names: List[str] = None,
    output_dir: str = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive evaluation dashboard with multiple plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        deltas: Delta scores
        features: Feature matrix
        coefficients: Calibration model coefficients
        novel_names: Novel name for each sample
        output_dir: Directory to save individual plots
        figsize: Figure size
        
    Returns:
        matplotlib Figure with all plots
    """
    fig = plt.figure(figsize=figsize, facecolor=COLORS['background'])
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Delta Distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['background'])
    
    deltas_arr = np.array(deltas)
    labels_arr = np.array(y_true)
    
    bins = np.linspace(min(deltas), max(deltas), 25)
    ax1.hist(deltas_arr[labels_arr == 1], bins=bins, alpha=0.7,
             color=COLORS['consistent'], label='Consistent', edgecolor='white')
    ax1.hist(deltas_arr[labels_arr == 0], bins=bins, alpha=0.7,
             color=COLORS['contradict'], label='Contradictory', edgecolor='white')
    ax1.axvline(x=0, color=COLORS['text'], linestyle='--', linewidth=1)
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Delta Distribution', fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    
    im = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.5 else COLORS['text']
            ax2.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                     ha='center', va='center', fontsize=10, color=color)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Contra', 'Consist'])
    ax2.set_yticklabels(['Contra', 'Consist'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix', fontweight='bold')
    
    # 3. Calibration Curve (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(COLORS['background'])
    
    y_prob_arr = np.array(y_prob)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(y_prob_arr, bin_edges) - 1, 0, 9)
    bin_sums = np.bincount(bin_indices, weights=np.array(y_true), minlength=10)
    bin_counts = np.maximum(np.bincount(bin_indices, minlength=10), 1)
    bin_freqs = bin_sums / bin_counts
    
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax3.plot(bin_centers, bin_freqs, 'o-', color=COLORS['neutral'], label='Model')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Actual Fraction')
    ax3.set_title('Calibration Curve', fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. Feature Scatter (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(COLORS['background'])
    
    ax4.scatter(features[labels_arr == 0, 0], features[labels_arr == 0, 2],
                c=COLORS['contradict'], alpha=0.5, s=20, label='Contradict')
    ax4.scatter(features[labels_arr == 1, 0], features[labels_arr == 1, 2],
                c=COLORS['consistent'], alpha=0.5, s=20, label='Consistent')
    ax4.set_xlabel('Delta Mean')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Feature Space', fontweight='bold')
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature Importance (bottom-center)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(COLORS['background'])
    
    if coefficients is not None:
        coef = np.array(coefficients).flatten()
        names = ['delta_mean', 'delta_max', 'cosine_mean', 'retrieval']
        colors = [COLORS['consistent'] if c > 0 else COLORS['contradict'] for c in coef]
        ax5.barh(range(len(coef)), coef, color=colors, edgecolor='white')
        ax5.set_yticks(range(len(names)))
        ax5.set_yticklabels(names)
        ax5.axvline(x=0, color=COLORS['text'], linewidth=0.8)
        ax5.set_xlabel('Coefficient')
        ax5.set_title('Feature Importance', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
    else:
        ax5.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax5.set_title('Feature Importance', fontweight='bold')
    
    # 6. Metrics Summary (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(COLORS['background'])
    ax6.axis('off')
    
    # Calculate metrics
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
    fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    
    metrics_text = f"""
    EVALUATION METRICS
    ══════════════════
    
    Accuracy:   {accuracy:.4f}
    Precision:  {precision:.4f}
    Recall:     {recall:.4f}
    F1 Score:   {f1:.4f}
    
    Samples:    {len(y_true)}
    Consistent: {sum(y_true)}
    Contradict: {len(y_true) - sum(y_true)}
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', color=COLORS['text'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax6.set_title('Summary', fontweight='bold')
    
    fig.suptitle('Generative Reasoning Evaluation Dashboard', 
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
    
    if output_dir:
        output_path = Path(output_dir) / 'evaluation_dashboard.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"✓ Dashboard saved: {output_path}")
    
    return fig


# ============================================================
# Exports
# ============================================================

__all__ = [
    'plot_delta_distribution',
    'plot_feature_scatter',
    'plot_calibration_curve',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_per_novel_scores',
    'create_evaluation_dashboard',
    'COLORS'
]
