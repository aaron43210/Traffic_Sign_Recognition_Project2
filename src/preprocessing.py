"""
Preprocessing Module
====================
Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) and
other preprocessing transforms for the GTSRB dataset.

CLAHE is essential for traffic sign recognition because signs are captured
under highly variable lighting conditions (shadows, glare, night, fog).
Operating on the L-channel of LAB color space preserves color information
while enhancing luminance contrast.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    IMG_SIZE, DATASET_MEAN, DATASET_STD, FIGURES_DIR
)
from src.utils import save_figure


class CLAHETransform:
    """
    Custom torchvision-compatible transform that applies CLAHE.
    
    Converts RGB → LAB, applies CLAHE to the L (luminance) channel,
    then converts back to RGB. This enhances local contrast without
    distorting color information.
    
    Parameters
    ----------
    clip_limit : float
        Threshold for contrast limiting (default: 2.0)
    tile_grid_size : tuple
        Size of grid for histogram equalization (default: (8,8))
    """
    
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """
        Apply CLAHE to a PIL Image.
        
        Parameters
        ----------
        img : PIL.Image
            Input RGB image
            
        Returns
        -------
        PIL.Image
            CLAHE-enhanced RGB image
        """
        # Convert PIL → NumPy (RGB)
        img_np = np.array(img)
        
        # Handle grayscale
        if len(img_np.shape) == 2:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
            img_np = clahe.apply(img_np)
            return Image.fromarray(img_np)
        
        # Convert RGB → LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert LAB → RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    def __repr__(self):
        return (f"CLAHETransform(clip_limit={self.clip_limit}, "
                f"tile_grid_size={self.tile_grid_size})")


class GlobalHistogramEqualization:
    """
    Standard global histogram equalization for comparison with CLAHE.
    Operates on the V channel in HSV color space.
    """
    
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            return Image.fromarray(cv2.equalizeHist(img_np))
        
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced)
    
    def __repr__(self):
        return "GlobalHistogramEqualization()"


def compute_dataset_statistics(dataloader):
    """
    Compute per-channel mean and std of the dataset for normalization.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader yielding (images, labels) batches
        
    Returns
    -------
    tuple
        (mean, std) each as numpy arrays of shape (3,)
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size
    
    mean /= n_samples
    std /= n_samples
    
    print(f"[✓] Dataset statistics computed over {n_samples} images:")
    print(f"    Mean: ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"    Std:  ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    
    return mean.numpy(), std.numpy()


def visualize_preprocessing_comparison(dataset, num_samples=8):
    """
    Create side-by-side comparison of original vs. preprocessed images.
    Shows: Original | Global HE | CLAHE
    
    Parameters
    ----------
    dataset : torchvision dataset
        Raw GTSRB dataset (without transforms)
    num_samples : int
        Number of sample images to display
    """
    clahe_transform = CLAHETransform()
    global_he = GlobalHistogramEqualization()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    fig.suptitle("Preprocessing Comparison: Original vs Global HE vs CLAHE",
                 fontsize=16, fontweight='bold', y=1.02)
    
    column_titles = ["Original", "Global Hist. Equalization", "CLAHE"]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_he = global_he(img_resized)
        img_clahe = clahe_transform(img_resized)
        
        axes[i, 0].imshow(img_resized)
        axes[i, 1].imshow(img_he)
        axes[i, 2].imshow(img_clahe)
        
        for j in range(3):
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f"Class {label}", fontsize=10, rotation=0,
                              labelpad=50, va='center')
    
    plt.tight_layout()
    save_figure(fig, "preprocessing_comparison.png")
    return fig


def visualize_pixel_distributions(dataset, num_samples=500):
    """
    Plot pixel intensity histograms before and after CLAHE.
    
    Parameters
    ----------
    dataset : torchvision dataset
        Raw GTSRB dataset
    num_samples : int
        Number of images to sample for histogram
    """
    clahe_transform = CLAHETransform()
    
    original_intensities = []
    clahe_intensities = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        img, _ = dataset[idx]
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_clahe = clahe_transform(img_resized)
        
        original_intensities.extend(np.array(img_resized).flatten().tolist())
        clahe_intensities.extend(np.array(img_clahe).flatten().tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pixel Intensity Distribution: Before vs After CLAHE",
                 fontsize=14, fontweight='bold')
    
    axes[0].hist(original_intensities, bins=256, range=(0, 255),
                 color='steelblue', alpha=0.7, density=True)
    axes[0].set_title("Original Images", fontsize=12)
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Density")
    
    axes[1].hist(clahe_intensities, bins=256, range=(0, 255),
                 color='coral', alpha=0.7, density=True)
    axes[1].set_title("After CLAHE", fontsize=12)
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Density")
    
    plt.tight_layout()
    save_figure(fig, "pixel_distribution_comparison.png")
    return fig
