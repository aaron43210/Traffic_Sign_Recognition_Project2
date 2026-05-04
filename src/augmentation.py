"""
Data Augmentation Module
========================
Defines training and testing transform pipelines for the GTSRB dataset.
Training transforms include extensive augmentation to simulate real-world
variations (lighting, viewpoint, occlusion). Test transforms apply only
the necessary preprocessing (CLAHE, resize, normalize).
"""

from torchvision import transforms

from src.config import (
    IMG_SIZE, DATASET_MEAN, DATASET_STD,
    ROTATION_DEGREES, TRANSLATE_RANGE, SCALE_RANGE, SHEAR_DEGREES,
    BRIGHTNESS, CONTRAST, SATURATION, HUE,
    PERSPECTIVE_DISTORTION, RANDOM_ERASING_PROB
)
from src.preprocessing import CLAHETransform


def get_train_transforms():
    """
    Build the training augmentation pipeline.
    
    Includes:
    - CLAHE for contrast enhancement
    - Geometric augmentations (rotation, affine, perspective)
    - Photometric augmentations (color jitter, gaussian blur)
    - Random erasing (simulates partial occlusion)
    - Normalization with dataset mean/std
    
    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        # 1. Resize to target size
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        
        # 2. CLAHE contrast enhancement
        CLAHETransform(),
        
        # 3. Geometric augmentations
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.RandomAffine(
            degrees=0,
            translate=TRANSLATE_RANGE,
            scale=SCALE_RANGE,
            shear=(-SHEAR_DEGREES, SHEAR_DEGREES,
                   -SHEAR_DEGREES, SHEAR_DEGREES),
        ),
        transforms.RandomPerspective(
            distortion_scale=PERSPECTIVE_DISTORTION,
            p=0.3
        ),
        
        # 4. Photometric augmentations
        transforms.ColorJitter(
            brightness=BRIGHTNESS,
            contrast=CONTRAST,
            saturation=SATURATION,
            hue=HUE
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # 5. Convert to tensor
        transforms.ToTensor(),
        
        # 6. Normalize using dataset statistics
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        
        # 7. Random erasing (simulates partial occlusion, operates on tensor)
        transforms.RandomErasing(
            p=RANDOM_ERASING_PROB,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        ),
    ])


def get_test_transforms():
    """
    Build the test/validation preprocessing pipeline.
    
    Only applies deterministic transforms:
    - Resize
    - CLAHE
    - ToTensor
    - Normalize
    
    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_no_clahe_test_transforms():
    """
    Test transforms WITHOUT CLAHE — for comparison experiments.
    
    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def visualize_augmentations(dataset, num_images=5, num_augmented=8):
    """
    Visualize original images alongside multiple augmented versions.
    
    Parameters
    ----------
    dataset : torchvision dataset
        Raw dataset (PIL images)
    num_images : int
        Number of original images to show
    num_augmented : int
        Number of augmented versions per image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from src.config import CLASS_NAMES, FIGURES_DIR
    from src.utils import save_figure
    
    train_transform = get_train_transforms()
    
    fig, axes = plt.subplots(num_images, num_augmented + 1,
                             figsize=(2.5 * (num_augmented + 1), 2.5 * num_images))
    fig.suptitle("Data Augmentation Visualization\n(Original + Augmented Versions)",
                 fontsize=14, fontweight='bold', y=1.02)
    
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Show original (column 0)
        if hasattr(img, 'numpy'):
            display_img = img.permute(1, 2, 0).numpy()
            display_img = np.clip(display_img, 0, 1)
        else:
            display_img = np.array(img) / 255.0 if np.array(img).max() > 1 else np.array(img)
        
        axes[i, 0].imshow(display_img)
        axes[i, 0].set_title(f"Original\n{CLASS_NAMES.get(label, label)[:20]}",
                             fontsize=8)
        axes[i, 0].axis('off')
        
        # Show augmented versions (columns 1+)
        for j in range(num_augmented):
            try:
                from PIL import Image
                if isinstance(img, torch.Tensor):
                    pil_img = transforms.ToPILImage()(img)
                else:
                    pil_img = img
                aug_img = train_transform(pil_img)
                
                # Denormalize for display
                mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
                std = torch.tensor(DATASET_STD).view(3, 1, 1)
                display = aug_img * std + mean
                display = display.permute(1, 2, 0).numpy()
                display = np.clip(display, 0, 1)
                
                axes[i, j + 1].imshow(display)
            except Exception:
                axes[i, j + 1].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f"Aug #{j+1}", fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, "augmentation_visualization.png")
    return fig


# Need this import for the visualize function
import torch
from src.config import DATASET_MEAN, DATASET_STD
