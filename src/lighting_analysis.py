"""
Lighting Condition Analysis Module
====================================
Evaluates model robustness under varying lighting conditions by
applying synthetic degradations to test images:
- Brightness adjustments (simulate day/night/overexposure)
- Contrast adjustments
- Gaussian noise (sensor noise)
- Simulated fog (additive haze)
- Simulated night conditions (extreme darkening + noise)

Produces accuracy-vs-degradation curves and per-class robustness scores.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import defaultdict

from src.config import (
    DEVICE, NUM_CLASSES, CLASS_NAMES, IMG_SIZE, BATCH_SIZE,
    BRIGHTNESS_FACTORS, CONTRAST_FACTORS, NOISE_SIGMAS,
    DATASET_MEAN, DATASET_STD, DATA_DIR, FIGURES_DIR,
    SAFETY_CRITICAL_CLASSES
)
from src.preprocessing import CLAHETransform
from src.utils import save_figure, create_plot_style


class LightingDegradation:
    """Apply various lighting degradations to PIL images."""
    
    @staticmethod
    def adjust_brightness(img, factor):
        """Adjust brightness by a multiplicative factor."""
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_contrast(img, factor):
        """Adjust contrast by a multiplicative factor."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def add_gaussian_noise(img, sigma):
        """Add Gaussian noise with given standard deviation."""
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def simulate_fog(img, intensity=0.5):
        """Simulate fog by blending with white."""
        img_np = np.array(img).astype(np.float32)
        white = np.ones_like(img_np) * 255.0
        foggy = img_np * (1 - intensity) + white * intensity
        return Image.fromarray(np.clip(foggy, 0, 255).astype(np.uint8))
    
    @staticmethod
    def simulate_night(img, darkness=0.3, noise_sigma=15):
        """Simulate night by darkening and adding noise."""
        darkened = LightingDegradation.adjust_brightness(img, darkness)
        return LightingDegradation.add_gaussian_noise(darkened, noise_sigma)


class DegradedDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset and applies a specific degradation before the standard transform.
    """
    
    def __init__(self, base_dataset, degradation_fn, transform):
        self.base_dataset = base_dataset
        self.degradation_fn = degradation_fn
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Apply degradation (expects PIL image)
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        img = self.degradation_fn(img)
        
        # Apply standard preprocessing
        if self.transform:
            img = self.transform(img)
        
        return img, label


def evaluate_under_condition(model, degradation_fn, condition_name,
                              use_clahe=True, device=DEVICE):
    """
    Evaluate model accuracy under a specific lighting condition.
    
    Parameters
    ----------
    model : nn.Module
    degradation_fn : callable
        Function that takes PIL image and returns degraded PIL image
    condition_name : str
    use_clahe : bool
        Whether to apply CLAHE after degradation
    device : torch.device
    
    Returns
    -------
    dict
        {overall_acc, per_class_acc}
    """
    # Build transform pipeline
    transform_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if use_clahe:
        transform_list.append(CLAHETransform())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])
    test_transform = transforms.Compose(transform_list)
    
    # Load raw test dataset
    raw_test = datasets.GTSRB(root=DATA_DIR, split='test', download=True,
                               transform=transforms.Resize((IMG_SIZE, IMG_SIZE)))
    
    # Create degraded dataset
    degraded = DegradedDataset(raw_test, degradation_fn, test_transform)
    loader = DataLoader(degraded, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for pred, true_label in zip(predicted.cpu(), labels.cpu()):
                cls = true_label.item()
                class_total[cls] += 1
                if pred.item() == cls:
                    class_correct[cls] += 1
    
    overall_acc = correct / total if total > 0 else 0
    per_class_acc = {}
    for cls_id in range(NUM_CLASSES):
        if class_total[cls_id] > 0:
            per_class_acc[cls_id] = class_correct[cls_id] / class_total[cls_id]
        else:
            per_class_acc[cls_id] = 0.0
    
    return {
        'overall_acc': overall_acc,
        'per_class_acc': per_class_acc,
        'condition': condition_name,
    }


def run_lighting_analysis(model, device=DEVICE):
    """
    Run complete lighting robustness analysis.
    
    Tests model under multiple brightness, contrast, noise, and weather conditions.
    Compares performance with and without CLAHE preprocessing.
    
    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate
    device : torch.device
    
    Returns
    -------
    dict
        Complete results for all conditions
    """
    model = model.to(device)
    results = {}
    
    print("\n" + "=" * 60)
    print("  LIGHTING ROBUSTNESS ANALYSIS")
    print("=" * 60)
    
    # 1. Brightness variations
    print("\n[1/5] Testing brightness variations...")
    brightness_results = {}
    for factor in BRIGHTNESS_FACTORS:
        fn = lambda img, f=factor: LightingDegradation.adjust_brightness(img, f)
        name = f"brightness_{factor}"
        r = evaluate_under_condition(model, fn, name, use_clahe=True, device=device)
        brightness_results[factor] = r['overall_acc']
        print(f"      Brightness {factor:.1f}x → Accuracy: {r['overall_acc']*100:.2f}%")
    results['brightness'] = brightness_results
    
    # 2. Contrast variations
    print("\n[2/5] Testing contrast variations...")
    contrast_results = {}
    for factor in CONTRAST_FACTORS:
        fn = lambda img, f=factor: LightingDegradation.adjust_contrast(img, f)
        name = f"contrast_{factor}"
        r = evaluate_under_condition(model, fn, name, use_clahe=True, device=device)
        contrast_results[factor] = r['overall_acc']
        print(f"      Contrast {factor:.1f}x → Accuracy: {r['overall_acc']*100:.2f}%")
    results['contrast'] = contrast_results
    
    # 3. Noise variations
    print("\n[3/5] Testing noise robustness...")
    noise_results = {}
    for sigma in NOISE_SIGMAS:
        fn = lambda img, s=sigma: LightingDegradation.add_gaussian_noise(img, s)
        name = f"noise_sigma_{sigma}"
        r = evaluate_under_condition(model, fn, name, use_clahe=True, device=device)
        noise_results[sigma] = r['overall_acc']
        print(f"      Noise σ={sigma:3d} → Accuracy: {r['overall_acc']*100:.2f}%")
    results['noise'] = noise_results
    
    # 4. CLAHE vs no-CLAHE comparison under low brightness
    print("\n[4/5] Testing CLAHE effectiveness under poor lighting...")
    clahe_comparison = {}
    for factor in [0.3, 0.5, 0.7, 1.0]:
        fn = lambda img, f=factor: LightingDegradation.adjust_brightness(img, f)
        
        with_clahe = evaluate_under_condition(model, fn, f"clahe_b{factor}",
                                               use_clahe=True, device=device)
        without_clahe = evaluate_under_condition(model, fn, f"no_clahe_b{factor}",
                                                  use_clahe=False, device=device)
        clahe_comparison[factor] = {
            'with_clahe': with_clahe['overall_acc'],
            'without_clahe': without_clahe['overall_acc'],
        }
        print(f"      Brightness {factor:.1f}x → With CLAHE: {with_clahe['overall_acc']*100:.2f}% | "
              f"Without: {without_clahe['overall_acc']*100:.2f}%")
    results['clahe_comparison'] = clahe_comparison
    
    # 5. Extreme conditions
    print("\n[5/5] Testing extreme conditions...")
    extreme_results = {}
    
    # Fog
    for intensity in [0.2, 0.4, 0.6]:
        fn = lambda img, i=intensity: LightingDegradation.simulate_fog(img, i)
        r = evaluate_under_condition(model, fn, f"fog_{intensity}", device=device)
        extreme_results[f'fog_{intensity}'] = r['overall_acc']
        print(f"      Fog intensity {intensity:.1f} → Accuracy: {r['overall_acc']*100:.2f}%")
    
    # Night
    fn = lambda img: LightingDegradation.simulate_night(img, darkness=0.3, noise_sigma=15)
    r = evaluate_under_condition(model, fn, "night", device=device)
    extreme_results['night'] = r['overall_acc']
    print(f"      Night simulation → Accuracy: {r['overall_acc']*100:.2f}%")
    
    results['extreme'] = extreme_results
    
    print("\n[✓] Lighting analysis complete!")
    return results


def plot_lighting_results(results):
    """
    Create comprehensive visualization of lighting analysis results.
    """
    create_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Robustness Under Varying Lighting Conditions",
                 fontsize=16, fontweight='bold')
    
    # 1. Brightness curve
    if 'brightness' in results:
        factors = sorted(results['brightness'].keys())
        accs = [results['brightness'][f] * 100 for f in factors]
        axes[0, 0].plot(factors, accs, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].fill_between(factors, accs, alpha=0.1, color='blue')
        axes[0, 0].set_xlabel("Brightness Factor")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_title("Brightness Robustness")
        axes[0, 0].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Normal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Contrast curve
    if 'contrast' in results:
        factors = sorted(results['contrast'].keys())
        accs = [results['contrast'][f] * 100 for f in factors]
        axes[0, 1].plot(factors, accs, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].fill_between(factors, accs, alpha=0.1, color='red')
        axes[0, 1].set_xlabel("Contrast Factor")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_title("Contrast Robustness")
        axes[0, 1].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Normal')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Noise curve
    if 'noise' in results:
        sigmas = sorted(results['noise'].keys())
        accs = [results['noise'][s] * 100 for s in sigmas]
        axes[1, 0].plot(sigmas, accs, 'go-', linewidth=2, markersize=8)
        axes[1, 0].fill_between(sigmas, accs, alpha=0.1, color='green')
        axes[1, 0].set_xlabel("Noise σ")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].set_title("Noise Robustness")
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. CLAHE comparison
    if 'clahe_comparison' in results:
        factors = sorted(results['clahe_comparison'].keys())
        with_clahe = [results['clahe_comparison'][f]['with_clahe'] * 100 for f in factors]
        without_clahe = [results['clahe_comparison'][f]['without_clahe'] * 100 for f in factors]
        
        x = np.arange(len(factors))
        width = 0.35
        axes[1, 1].bar(x - width/2, with_clahe, width, label='With CLAHE',
                        color='#4ECDC4', edgecolor='white')
        axes[1, 1].bar(x + width/2, without_clahe, width, label='Without CLAHE',
                        color='#FF6B6B', edgecolor='white')
        axes[1, 1].set_xlabel("Brightness Factor")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].set_title("CLAHE Effectiveness Under Low Light")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"{f:.1f}x" for f in factors])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "lighting_robustness_analysis.png")
    return fig


def plot_degradation_samples(num_samples=4):
    """
    Visualize sample images under different degradation conditions.
    """
    create_plot_style()
    
    raw_test = datasets.GTSRB(root=DATA_DIR, split='test', download=True,
                               transform=transforms.Resize((IMG_SIZE, IMG_SIZE)))
    
    degradations = {
        'Original': lambda img: img,
        'Dark (0.3x)': lambda img: LightingDegradation.adjust_brightness(img, 0.3),
        'Bright (2.0x)': lambda img: LightingDegradation.adjust_brightness(img, 2.0),
        'Low Contrast': lambda img: LightingDegradation.adjust_contrast(img, 0.3),
        'Noisy (σ=30)': lambda img: LightingDegradation.add_gaussian_noise(img, 30),
        'Fog (0.5)': lambda img: LightingDegradation.simulate_fog(img, 0.5),
        'Night': lambda img: LightingDegradation.simulate_night(img),
    }
    
    indices = np.random.choice(len(raw_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, len(degradations),
                             figsize=(2.5 * len(degradations), 2.5 * num_samples))
    fig.suptitle("Synthetic Lighting Degradation Samples",
                 fontsize=14, fontweight='bold', y=1.02)
    
    for j, (name, fn) in enumerate(degradations.items()):
        axes[0, j].set_title(name, fontsize=9, fontweight='bold')
    
    for i, idx in enumerate(indices):
        img, label = raw_test[idx]
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        for j, (name, fn) in enumerate(degradations.items()):
            degraded = fn(img)
            axes[i, j].imshow(np.array(degraded))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(f"Class {label}", fontsize=8, rotation=0,
                                      labelpad=40, va='center')
    
    plt.tight_layout()
    save_figure(fig, "degradation_samples.png")
    return fig
