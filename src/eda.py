"""
EDA (Exploratory Data Analysis) Module
=======================================
Comprehensive visualization functions for understanding the GTSRB dataset:
class distribution, image properties, brightness analysis, and sample grids.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from collections import Counter

from src.config import (
    CLASS_NAMES, SIGN_CATEGORIES, NUM_CLASSES,
    FIGURES_DIR, FIGURE_DPI, IMG_SIZE
)
from src.utils import save_figure, create_plot_style


def plot_class_distribution(dataset, title="GTSRB Class Distribution"):
    """
    Plot the distribution of samples across all 43 classes.
    Highlights the severe class imbalance in GTSRB.
    
    Parameters
    ----------
    dataset : torch Dataset
        GTSRB dataset
    title : str
        Plot title
    """
    create_plot_style()
    
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Color by sign category
    colors = []
    category_colors = {
        "Speed Limits": '#FF6B6B',
        "Prohibitory": '#4ECDC4',
        "Mandatory": '#45B7D1',
        "Danger/Warning": '#FFA07A',
        "Priority/Stop": '#98D8C8',
        "End of Restriction": '#C9B1FF',
    }
    
    for cls_id in classes:
        color = '#AAAAAA'  # default
        for cat_name, cat_ids in SIGN_CATEGORIES.items():
            if cls_id in cat_ids:
                color = category_colors.get(cat_name, '#AAAAAA')
                break
        colors.append(color)
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Class ID", fontsize=13)
    ax.set_ylabel("Number of Samples", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, rotation=45, fontsize=8)
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=name)
                       for name, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title="Sign Category", title_fontsize=11)
    
    # Add imbalance annotation
    max_count = max(counts)
    min_count = min(counts)
    ax.annotate(f"Imbalance Ratio: {max_count/min_count:.1f}x\n"
                f"Max: {max_count} (Class {classes[counts.index(max_count)]})\n"
                f"Min: {min_count} (Class {classes[counts.index(min_count)]})",
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, "class_distribution.png")
    return fig


def plot_sign_category_distribution(dataset):
    """
    Plot distribution grouped by sign category (pie + bar chart).
    """
    create_plot_style()
    
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    
    category_counts = {}
    for cat_name, cat_ids in SIGN_CATEGORIES.items():
        category_counts[cat_name] = sum(counter.get(c, 0) for c in cat_ids)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sign Category Distribution", fontsize=15, fontweight='bold')
    
    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C9B1FF']
    axes[0].pie(category_counts.values(), labels=category_counts.keys(),
                colors=colors, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[0].set_title("Proportion by Category", fontsize=12)
    
    # Bar chart
    axes[1].barh(list(category_counts.keys()), list(category_counts.values()),
                 color=colors, edgecolor='white')
    axes[1].set_xlabel("Number of Samples")
    axes[1].set_title("Count by Category", fontsize=12)
    
    for i, (cat, count) in enumerate(category_counts.items()):
        axes[1].text(count + 50, i, str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "category_distribution.png")
    return fig


def plot_sample_grid(dataset, samples_per_class=3):
    """
    Display a grid showing sample images from all 43 classes.
    
    Parameters
    ----------
    dataset : torch Dataset
        GTSRB dataset (should return tensors)
    samples_per_class : int
        Number of sample images per class
    """
    create_plot_style()
    
    # Collect samples per class
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    
    for img, label in dataset:
        if len(class_samples[label]) < samples_per_class:
            if isinstance(img, torch.Tensor):
                class_samples[label].append(img)
            else:
                from torchvision import transforms
                class_samples[label].append(transforms.ToTensor()(img))
        
        # Check if we have enough
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break
    
    # Create grid
    n_cols = samples_per_class
    n_rows = NUM_CLASSES
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.5))
    fig.suptitle(f"GTSRB: All 43 Traffic Sign Classes ({samples_per_class} samples each)",
                 fontsize=16, fontweight='bold', y=1.01)
    
    for cls_id in range(NUM_CLASSES):
        for j in range(samples_per_class):
            ax = axes[cls_id, j]
            if j < len(class_samples[cls_id]):
                img = class_samples[cls_id][j]
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                ax.imshow(img_np)
            ax.axis('off')
            
            if j == 0:
                short_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                if len(short_name) > 25:
                    short_name = short_name[:22] + "..."
                ax.set_ylabel(f"{cls_id}: {short_name}", fontsize=6,
                              rotation=0, labelpad=100, va='center')
    
    plt.tight_layout()
    save_figure(fig, "sample_grid_all_classes.png")
    return fig


def plot_brightness_analysis(dataset, num_samples=1000):
    """
    Analyze mean brightness distribution across classes.
    Some classes are systematically darker due to capture conditions.
    
    Parameters
    ----------
    dataset : torch Dataset
        GTSRB dataset (returns tensors)
    num_samples : int
        Number of images to analyze
    """
    create_plot_style()
    
    class_brightness = {i: [] for i in range(NUM_CLASSES)}
    count = 0
    
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            brightness = img.mean().item()
        else:
            brightness = np.array(img).mean() / 255.0
        class_brightness[label].append(brightness)
        count += 1
        if count >= num_samples:
            break
    
    # Compute stats per class
    classes = sorted(class_brightness.keys())
    means = [np.mean(class_brightness[c]) if class_brightness[c] else 0 for c in classes]
    stds = [np.std(class_brightness[c]) if class_brightness[c] else 0 for c in classes]
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("Brightness Analysis Across Classes", fontsize=15, fontweight='bold')
    
    # Bar chart of mean brightness
    colors = ['#FF6B6B' if m < np.percentile(means, 25) else
              '#4ECDC4' if m > np.percentile(means, 75) else '#45B7D1'
              for m in means]
    
    axes[0].bar(classes, means, yerr=stds, color=colors, edgecolor='white',
                capsize=2, alpha=0.8)
    axes[0].set_xlabel("Class ID")
    axes[0].set_ylabel("Mean Pixel Intensity")
    axes[0].set_title("Mean Brightness per Class (red = darkest, green = brightest)")
    axes[0].set_xticks(classes)
    axes[0].set_xticklabels(classes, fontsize=7)
    axes[0].axhline(y=np.mean(means), color='red', linestyle='--', alpha=0.5,
                     label=f'Overall mean: {np.mean(means):.3f}')
    axes[0].legend()
    
    # Overall brightness histogram
    all_brightness = [b for cls_list in class_brightness.values() for b in cls_list]
    axes[1].hist(all_brightness, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel("Mean Pixel Intensity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Overall Brightness Distribution")
    axes[1].axvline(x=np.mean(all_brightness), color='red', linestyle='--',
                     label=f'Mean: {np.mean(all_brightness):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    save_figure(fig, "brightness_analysis.png")
    return fig


def plot_image_size_distribution(data_dir):
    """
    Plot the original image size distribution before resizing.
    GTSRB images have variable sizes from 15x15 to 250x250.
    
    Parameters
    ----------
    data_dir : str
        Path to GTSRB data directory
    """
    import os
    from PIL import Image as PILImage
    
    create_plot_style()
    
    widths, heights = [], []
    
    # Walk through the training directory to get original sizes
    train_dir = os.path.join(data_dir, "gtsrb", "GTSRB", "Training")
    if not os.path.exists(train_dir):
        # Try alternative path
        train_dir = os.path.join(data_dir, "GTSRB", "Training")
    
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            for f in files:
                if f.endswith('.ppm') or f.endswith('.png'):
                    try:
                        img = PILImage.open(os.path.join(root, f))
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                    except Exception:
                        continue
                    if len(widths) > 5000:
                        break
    
    if not widths:
        print("[!] Could not find original images for size analysis")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Original Image Size Distribution (Before Resizing)",
                 fontsize=14, fontweight='bold')
    
    axes[0].scatter(widths, heights, alpha=0.1, s=5, c='steelblue')
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].set_title(f"Width vs Height (n={len(widths)})")
    axes[0].plot([0, 300], [0, 300], 'r--', alpha=0.5, label='Square')
    axes[0].legend()
    
    axes[1].hist(widths, bins=50, alpha=0.6, label='Width', color='steelblue')
    axes[1].hist(heights, bins=50, alpha=0.6, label='Height', color='coral')
    axes[1].set_xlabel("Size (pixels)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Size Histogram")
    axes[1].legend()
    
    plt.tight_layout()
    save_figure(fig, "image_size_distribution.png")
    return fig


def generate_dataset_summary(dataset):
    """
    Print a comprehensive dataset summary.
    
    Parameters
    ----------
    dataset : torch Dataset
        GTSRB dataset
        
    Returns
    -------
    dict
        Summary statistics
    """
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    
    summary = {
        "total_samples": len(dataset),
        "num_classes": len(counter),
        "min_class_size": min(counter.values()),
        "max_class_size": max(counter.values()),
        "mean_class_size": np.mean(list(counter.values())),
        "std_class_size": np.std(list(counter.values())),
        "imbalance_ratio": max(counter.values()) / min(counter.values()),
        "smallest_class": min(counter, key=counter.get),
        "largest_class": max(counter, key=counter.get),
    }
    
    print("\n" + "=" * 60)
    print("  GTSRB Dataset Summary")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.2f}")
        else:
            print(f"  {key:.<40} {value}")
    print("=" * 60)
    
    return summary
