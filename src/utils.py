"""
Utility Functions
=================
Common helpers for reproducibility, plotting, timing, and device management.
"""

import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for notebook/server

from src.config import RANDOM_SEED, DEVICE, FIGURE_DPI, FIGURES_DIR


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[✓] Random seed set to {seed}")


def get_device():
    """Return the best available device and print info."""
    print(f"[✓] Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif DEVICE.type == 'mps':
        print("    Apple Silicon GPU (MPS)")
    return DEVICE


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.description:
            print(f"[⏱] {self.description}: {self.elapsed:.2f}s")


def save_figure(fig, filename: str, dpi: int = FIGURE_DPI):
    """Save a matplotlib figure to the figures directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[✓] Figure saved: {filepath}")
    return filepath


def count_parameters(model: torch.nn.Module) -> int:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[✓] Model parameters: {total:,} total | {trainable:,} trainable")
    return trainable


def format_metrics(metrics: dict, title: str = "Metrics") -> str:
    """Format a metrics dictionary as a readable string."""
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:.<35} {value:.4f}")
        else:
            lines.append(f"  {key:.<35} {value}")
    lines.append(f"{'='*50}\n")
    return "\n".join(lines)


def create_plot_style():
    """Apply professional plotting style."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': FIGURE_DPI,
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
