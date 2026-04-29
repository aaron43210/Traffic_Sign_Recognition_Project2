"""
Explainability Module
=====================
Model interpretation and explainability tools including:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Feature map visualization
- STN transformation visualization
- t-SNE embedding visualization
- Most/least confident prediction analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from src.config import (
    NUM_CLASSES, CLASS_NAMES, DEVICE, IMG_SIZE,
    DATASET_MEAN, DATASET_STD, FIGURES_DIR
)
from src.utils import save_figure, create_plot_style


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Produces a heatmap highlighting the regions of the input image
    that are most important for the model's prediction.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization", ICCV 2017.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Parameters
        ----------
        input_image : torch.Tensor
            Single image tensor (1, C, H, W)
        target_class : int or None
            Class to generate heatmap for. If None, uses predicted class.
            
        Returns
        -------
        tuple
            (heatmap_numpy, predicted_class, confidence)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(cam, size=(input_image.shape[2], input_image.shape[3]),
                            mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, confidence
    
    def cleanup(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model):
    """
    Automatically find the last Conv2d layer for Grad-CAM.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def visualize_gradcam(model, images, labels, device=DEVICE, num_samples=10):
    """
    Create Grad-CAM visualization grid.
    
    Parameters
    ----------
    model : nn.Module
    images : list of tensors or DataLoader
    labels : list of ints
    device : torch.device
    num_samples : int
    """
    create_plot_style()
    model = model.to(device)
    
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    
    mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
    std = torch.tensor(DATASET_STD).view(3, 1, 1)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples))
    fig.suptitle("Grad-CAM Visualization\n(Where does the model look?)",
                 fontsize=16, fontweight='bold', y=1.01)
    
    column_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold')
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        true_label = labels[i] if isinstance(labels[i], int) else labels[i].item()
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(device)
        
        # Generate Grad-CAM
        heatmap, pred_class, confidence = grad_cam.generate(img)
        
        # Denormalize for display
        img_display = img.squeeze().cpu() * std + mean
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        
        # Original image
        axes[i, 0].imshow(img_display)
        is_correct = pred_class == true_label
        color = 'green' if is_correct else 'red'
        axes[i, 0].set_ylabel(
            f"True: {true_label}\nPred: {pred_class}\nConf: {confidence:.1%}",
            fontsize=8, color=color, fontweight='bold',
            rotation=0, labelpad=60, va='center'
        )
        
        # Heatmap
        axes[i, 1].imshow(heatmap, cmap='jet', alpha=0.8)
        
        # Overlay
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        overlay = 0.5 * img_display + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        axes[i, 2].imshow(overlay)
        
        for j in range(3):
            axes[i, j].axis('off')
    
    grad_cam.cleanup()
    plt.tight_layout()
    save_figure(fig, "gradcam_visualization.png")
    return fig


def visualize_stn_transformation(model, images, labels, device=DEVICE, num_samples=8):
    """
    Visualize what the Spatial Transformer Network does to the input.
    Shows: Original → STN-transformed image side by side.
    """
    create_plot_style()
    model = model.to(device)
    model.eval()
    
    if not hasattr(model, 'get_stn_output'):
        print("[!] Model does not have STN component")
        return None
    
    mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
    std = torch.tensor(DATASET_STD).view(3, 1, 1)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2.5 * num_samples))
    fig.suptitle("Spatial Transformer Network Visualization\n(How STN rectifies the input)",
                 fontsize=14, fontweight='bold', y=1.02)
    
    axes[0, 0].set_title("Original Input", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("After STN Transform", fontsize=12, fontweight='bold')
    
    with torch.no_grad():
        for i in range(min(num_samples, len(images))):
            img = images[i]
            label = labels[i] if isinstance(labels[i], int) else labels[i].item()
            
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            
            # Get STN output
            stn_output = model.get_stn_output(img)
            
            # Denormalize
            orig = img.squeeze().cpu() * std + mean
            orig = orig.permute(1, 2, 0).numpy()
            orig = np.clip(orig, 0, 1)
            
            transformed = stn_output.squeeze().cpu() * std + mean
            transformed = transformed.permute(1, 2, 0).numpy()
            transformed = np.clip(transformed, 0, 1)
            
            axes[i, 0].imshow(orig)
            axes[i, 1].imshow(transformed)
            
            name = CLASS_NAMES.get(label, f"Class {label}")[:25]
            axes[i, 0].set_ylabel(f"{label}: {name}", fontsize=8,
                                  rotation=0, labelpad=80, va='center')
            
            for j in range(2):
                axes[i, j].axis('off')
    
    plt.tight_layout()
    save_figure(fig, "stn_transformation.png")
    return fig


def visualize_feature_maps(model, image, device=DEVICE):
    """
    Visualize intermediate feature maps at each convolutional block.
    """
    create_plot_style()
    model = model.to(device)
    model.eval()
    
    if not hasattr(model, 'get_feature_maps'):
        print("[!] Model does not support feature map extraction")
        return None
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        features = model.get_feature_maps(image)
    
    n_blocks = len(features)
    fig, axes = plt.subplots(n_blocks, 8, figsize=(20, 3 * n_blocks))
    fig.suptitle("Feature Maps at Each Convolutional Block",
                 fontsize=14, fontweight='bold', y=1.02)
    
    for i, (name, fmap) in enumerate(features.items()):
        fmap = fmap.squeeze().cpu().numpy()
        n_channels = min(8, fmap.shape[0])
        
        for j in range(8):
            if j < n_channels:
                axes[i, j].imshow(fmap[j], cmap='viridis')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=10, fontweight='bold',
                                      rotation=0, labelpad=60, va='center')
            if i == 0:
                axes[i, j].set_title(f"Ch {j}", fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, "feature_maps.png")
    return fig


def plot_tsne_embeddings(model, dataloader, device=DEVICE, n_samples=2000):
    """
    Visualize t-SNE of penultimate layer embeddings colored by class.
    """
    create_plot_style()
    
    from sklearn.manifold import TSNE
    
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels_list = []
    
    # Hook to capture penultimate layer output
    hook_output = []
    
    def hook_fn(module, input, output):
        hook_output.append(input[0].detach().cpu())
    
    # Register hook on the last linear layer
    last_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    
    if last_linear is None:
        print("[!] Could not find linear layer for embedding extraction")
        return None
    
    handle = last_linear.register_forward_hook(hook_fn)
    
    count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)
            
            embeddings.append(hook_output[-1])
            labels_list.extend(labels.numpy().tolist())
            
            count += len(labels)
            if count >= n_samples:
                break
    
    handle.remove()
    
    all_embeddings = torch.cat(embeddings, dim=0).numpy()[:n_samples]
    all_labels = np.array(labels_list[:n_samples])
    
    # Run t-SNE
    print("[...] Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=all_labels, cmap='tab20', s=8, alpha=0.6
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Class ID", fontsize=11)
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE Visualization of Learned Embeddings\n(Penultimate Layer)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_figure(fig, "tsne_embeddings.png")
    return fig
