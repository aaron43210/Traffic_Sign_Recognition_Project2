"""
Configuration Module
====================
Centralizes all hyperparameters, paths, constants, and class label mappings
for the GTSRB Traffic Sign Recognition project.
"""

import os
import torch

# Fix for STN on Mac (MPS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Create directories if they don't exist
for _dir in [DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ──────────────────────────────────────────────
# Dataset Parameters
# ──────────────────────────────────────────────
NUM_CLASSES = 43
IMG_SIZE = 48              # Resize all images to 48x48 (better detail than 32x32)
IMG_CHANNELS = 3           # RGB

# GTSRB dataset mean and std (computed over training set)
# These are widely-used values for the GTSRB dataset
DATASET_MEAN = (0.3403, 0.3121, 0.3214)
DATASET_STD = (0.2724, 0.2608, 0.2669)

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
VAL_SPLIT = 0.15          # 15% of training data for validation

# Scheduler
SCHEDULER_T0 = 10         # CosineAnnealingWarmRestarts T_0
SCHEDULER_TMULT = 2       # CosineAnnealingWarmRestarts T_mult

# ──────────────────────────────────────────────
# CLAHE Parameters
# ──────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ──────────────────────────────────────────────
# Augmentation Parameters
# ──────────────────────────────────────────────
ROTATION_DEGREES = 15
TRANSLATE_RANGE = (0.1, 0.1)
SCALE_RANGE = (0.9, 1.1)
SHEAR_DEGREES = 10
BRIGHTNESS = 0.3
CONTRAST = 0.3
SATURATION = 0.3
HUE = 0.1
PERSPECTIVE_DISTORTION = 0.2
RANDOM_ERASING_PROB = 0.1

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Dropout rates (progressive)
DROPOUT_CONV1 = 0.1
DROPOUT_CONV2 = 0.2
DROPOUT_CONV3 = 0.3
DROPOUT_FC = 0.5

# ──────────────────────────────────────────────
# GTSRB Class Labels (0–42)
# ──────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5t",
}

# Sign category groupings for analysis
SIGN_CATEGORIES = {
    "Speed Limits":     [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "Prohibitory":      [9, 10, 15, 16, 17],
    "Mandatory":        [33, 34, 35, 36, 37, 38, 39, 40],
    "Danger/Warning":   [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Priority/Stop":    [12, 13, 14],
    "End of Restriction": [6, 32, 41, 42],
}

# Safety-critical classes (misclassification could endanger lives)
SAFETY_CRITICAL_CLASSES = [
    14,  # Stop
    13,  # Yield
    17,  # No entry
    12,  # Priority road
    0, 1, 2, 3, 4, 5, 7, 8,  # Speed limits
    25,  # Road work
    28,  # Children crossing
    27,  # Pedestrians
    26,  # Traffic signals
]

# ──────────────────────────────────────────────
# Lighting Analysis Parameters
# ──────────────────────────────────────────────
BRIGHTNESS_FACTORS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.0]
CONTRAST_FACTORS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.0]
NOISE_SIGMAS = [0, 5, 10, 20, 30, 50]

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
FIGURE_DPI = 150
COLORMAP = "viridis"
RANDOM_SEED = 42
