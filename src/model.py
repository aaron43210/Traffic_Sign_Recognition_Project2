"""
Model Architecture Module
==========================
Defines three CNN architectures of increasing sophistication for GTSRB:

1. BaselineCNN      — Simple 3-block CNN (baseline comparison)
2. EnhancedCNN      — 6-layer CNN with progressive dropout + BN
3. STN_CNN          — Spatial Transformer Network + Enhanced CNN (primary model)

All models use batch normalization and dropout as required by the project spec.
The STN-CNN is the primary model, achieving >99% accuracy by learning to
spatially rectify input images before classification.

References:
    - Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015
    - Ciresan et al., "Multi-Column DNN for Traffic Sign Classification", 2012
    - Sermanet & LeCun, "Traffic Sign Recognition with Multi-Scale CNNs", 2011
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES, IMG_SIZE, IMG_CHANNELS


# ═══════════════════════════════════════════════════
# Model 1: Baseline CNN
# ═══════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    """
    Simple baseline CNN with 3 convolutional blocks.
    Uses BatchNorm and Dropout for regularization.
    
    Architecture:
        Conv(3→32, 3x3) → BN → ReLU → Conv(32→32, 3x3) → BN → ReLU → MaxPool → Drop(0.1)
        Conv(32→64, 3x3) → BN → ReLU → Conv(64→64, 3x3) → BN → ReLU → MaxPool → Drop(0.2)
        Conv(64→128, 3x3) → BN → ReLU → MaxPool → Drop(0.3)
        FC(128*4*4→256) → BN → ReLU → Drop(0.5) → FC(256→43)
    
    Expected accuracy: ~96-97%
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(BaselineCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # Compute the flattened size dynamically
        self._flat_size = self._get_flat_size()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def _get_flat_size(self):
        """Compute the flattened feature size after conv blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
            out = self.block1(dummy)
            out = self.block2(out)
            out = self.block3(out)
            return out.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════
# Model 2: Enhanced CNN (Deeper + Progressive Dropout)
# ═══════════════════════════════════════════════════

class EnhancedCNN(nn.Module):
    """
    Deeper CNN with 6 convolutional layers, progressive dropout,
    batch normalization, and global average pooling.
    
    Architecture:
        Block1: Conv(3→64, 3) → BN → ReLU → Conv(64→64, 3) → BN → ReLU → MaxPool → Drop2d(0.1)
        Block2: Conv(64→128, 3) → BN → ReLU → Conv(128→128, 3) → BN → ReLU → MaxPool → Drop2d(0.2)
        Block3: Conv(128→256, 3) → BN → ReLU → Conv(256→256, 3) → BN → ReLU → MaxPool → Drop2d(0.3)
        GAP → FC(256→512) → BN → ReLU → Drop(0.5) → FC(512→43)
    
    Expected accuracy: ~98-99%
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(EnhancedCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """Return intermediate feature maps for visualization."""
        features = {}
        x = self.block1(x)
        features['block1'] = x.clone()
        x = self.block2(x)
        features['block2'] = x.clone()
        x = self.block3(x)
        features['block3'] = x.clone()
        return features


# ═══════════════════════════════════════════════════
# Model 3: STN-CNN (Primary Model)
# ═══════════════════════════════════════════════════

class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network module.
    
    Learns to spatially transform (crop, rotate, scale) the input image
    to normalize the sign's position and orientation. This is critical for
    real-world traffic signs which appear at various viewpoints.
    
    The STN consists of:
    1. Localization Network: Small CNN that predicts the transformation parameters
    2. Grid Generator: Creates a sampling grid from the parameters
    3. Sampler: Applies the grid to transform the input image
    
    Reference: Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015
    """
    
    def __init__(self, in_channels=IMG_CHANNELS, img_size=IMG_SIZE):
        super(SpatialTransformerNetwork, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )
        
        # Initialize weights to identity transform
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        """
        Apply learned spatial transformation to input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (B, C, H, W)
            
        Returns
        -------
        torch.Tensor
            Spatially transformed images (same shape)
        """
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x


class STN_CNN(nn.Module):
    """
    Primary model: Spatial Transformer Network + Enhanced CNN.
    
    Combines the spatial invariance learning of STN with the
    classification power of the Enhanced CNN architecture.
    
    Architecture:
        STN → Enhanced CNN backbone
    
    The STN learns to crop, rotate, and scale the input image to
    normalize the traffic sign's appearance, while the CNN backbone
    extracts features and classifies.
    
    Expected accuracy: >99%
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 43)
    use_stn : bool
        Whether to use STN (can be disabled for ablation studies)
    """
    
    def __init__(self, num_classes=NUM_CLASSES, use_stn=True):
        super(STN_CNN, self).__init__()
        
        self.use_stn = use_stn
        
        # Spatial Transformer
        if use_stn:
            self.stn = SpatialTransformerNetwork()
        
        # Feature extractor (same as EnhancedCNN blocks)
        self.block1 = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        # Apply spatial transformation
        if self.use_stn:
            x = self.stn(x)
        
        # Extract features
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Pool and classify
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_stn_output(self, x):
        """
        Get the STN-transformed image for visualization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images
            
        Returns
        -------
        torch.Tensor
            STN-transformed images
        """
        if self.use_stn:
            return self.stn(x)
        return x
    
    def get_feature_maps(self, x):
        """Return intermediate feature maps for visualization."""
        features = {}
        
        if self.use_stn:
            x = self.stn(x)
            features['stn_output'] = x.clone()
        
        x = self.block1(x)
        features['block1'] = x.clone()
        x = self.block2(x)
        features['block2'] = x.clone()
        x = self.block3(x)
        features['block3'] = x.clone()
        
        return features
    
    def get_last_conv_layer(self):
        """Return the last convolutional layer (for Grad-CAM)."""
        # The last Conv2d in block3
        for layer in reversed(list(self.block3.children())):
            if isinstance(layer, nn.Conv2d):
                return layer
        return self.block3[-4]  # Fallback: second Conv2d in block3


def build_model(model_type='stn_cnn', **kwargs):
    """
    Factory function to build models by name.
    
    Parameters
    ----------
    model_type : str
        One of 'baseline', 'enhanced', 'stn_cnn'
    **kwargs
        Additional arguments passed to model constructor
        
    Returns
    -------
    nn.Module
    """
    models = {
        'baseline': BaselineCNN,
        'enhanced': EnhancedCNN,
        'stn_cnn': STN_CNN,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from {list(models.keys())}")
    
    model = models[model_type](**kwargs)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[✓] Built model: {model_type}")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    return model


def model_summary(model, input_size=(3, IMG_SIZE, IMG_SIZE)):
    """
    Print a detailed model summary similar to Keras model.summary().
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    input_size : tuple
        Input tensor shape (C, H, W)
    """
    print("\n" + "=" * 70)
    print(f"{'Layer':.<40} {'Output Shape':.<20} {'Params'}")
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                               nn.BatchNorm1d, nn.Dropout, nn.Dropout2d)):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            if any(p.requires_grad for p in module.parameters()):
                trainable_params += params
            
            print(f"  {name:.<38} {'—':.<20} {params:,}")
    
    print("=" * 70)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Non-trainable:    {total_params - trainable_params:,}")
    print("=" * 70)
