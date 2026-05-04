"""
Traffic Sign Recognition — Streamlit Deployment
=================================================
Interactive web application for real-time traffic sign classification.

Features:
- Image upload (JPG/PNG/JPEG)
- Real-time prediction with top-5 class probabilities
- Confidence gauge visualization
- Grad-CAM attention overlay
- Sign information panel
- Lighting simulation (brightness/contrast sliders)
- Sample gallery of test images
"""

import os
import sys

# ─────────────────────────────────────────
# Path Configuration (MUST BE FIRST)
# ─────────────────────────────────────────
# Add project root to path to ensure 'src' is importable on all platforms
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.config import (
    NUM_CLASSES, CLASS_NAMES, IMG_SIZE,
    DATASET_MEAN, DATASET_STD, MODEL_DIR, SIGN_CATEGORIES,
    SAFETY_CRITICAL_CLASSES
)
from src.preprocessing import CLAHETransform
from src.model import STN_CNN, EnhancedCNN, BaselineCNN
from src.explainability import GradCAM, get_target_layer
from torchvision import transforms


# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition — GTSRB",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .safety-warning {
        background: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .safety-critical {
        background: #F8D7DA;
        border-left: 4px solid #DC3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name="stn_cnn"):
    """Load trained model from checkpoint."""
    model_map = {
        'stn_cnn': STN_CNN,
        'enhanced': EnhancedCNN,
        'baseline': BaselineCNN,
    }
    
    model_class = model_map.get(model_name, STN_CNN)
    model = model_class(num_classes=NUM_CLASSES)
    
    # Robust path detection for Streamlit Cloud
    # Priority: 1. Current directory/models, 2. MODEL_DIR from config
    filename = "baseline_model.pth" if model_name == "baseline" else f"{model_name}_best.pth"
    
    possible_paths = [
        os.path.join(ROOT_DIR, "models", filename),
        os.path.join(MODEL_DIR, filename),
        os.path.abspath(os.path.join("models", filename))
    ]
    
    checkpoint_path = None
    for p in possible_paths:
        if os.path.exists(p):
            checkpoint_path = p
            break
    
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # Handle both full checkpoints and state_dicts
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            
            val_acc = checkpoint.get('val_acc', 'N/A')
            if isinstance(val_acc, (int, float)):
                st.sidebar.success(f"✅ Loaded {model_name} (Val Acc: {val_acc:.2f}%)")
            else:
                st.sidebar.success(f"✅ Loaded {model_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {model_name}: {str(e)}")
    else:
        st.sidebar.warning(f"⚠️ No checkpoint found for {model_name}. Tried: {possible_paths[0]}")
        st.sidebar.info("Falling back to random weights (low confidence expected).")
    
    model.eval()
    return model


def preprocess_image(image, use_clahe=True):
    """Preprocess uploaded image for model input."""
    transform_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if use_clahe:
        transform_list.append(CLAHETransform())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])
    transform = transforms.Compose(transform_list)
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)


def generate_gradcam(model, input_tensor):
    """Generate Grad-CAM heatmap for the prediction."""
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    heatmap, pred_class, confidence = grad_cam.generate(input_tensor)
    grad_cam.cleanup()
    return heatmap, pred_class, confidence


def get_sign_category(class_id):
    """Get the category name for a traffic sign class."""
    for cat_name, cat_ids in SIGN_CATEGORIES.items():
        if class_id in cat_ids:
            return cat_name
    return "Other"


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["stn_cnn", "enhanced", "baseline"],
    index=0,
    format_func=lambda x: {
        'stn_cnn': '🌟 STN-CNN (Best)',
        'enhanced': '📊 Enhanced CNN',
        'baseline': '📈 Baseline CNN',
    }[x]
)

use_clahe = st.sidebar.checkbox("Apply CLAHE Preprocessing", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔆 Lighting Simulation")
brightness = st.sidebar.slider("Brightness", 0.2, 3.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.2, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
This app classifies **43 categories** of German traffic signs
using a CNN with **Spatial Transformer Network**.

**Dataset:** GTSRB  
**Architecture:** STN + CNN with BatchNorm & Dropout  
**Preprocessing:** CLAHE histogram equalization
""")

# ─────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────
st.markdown('<p class="main-header">🚦 Traffic Sign Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classify 43 categories of traffic signs for autonomous vehicles</p>',
            unsafe_allow_html=True)

# Load model
model = load_model(model_choice)

# Upload section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Traffic Sign Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a traffic sign image for classification"
    )

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Apply lighting simulation
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess and predict
        input_tensor = preprocess_image(image, use_clahe=use_clahe)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().numpy()
        
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        with col2:
            # Prediction result
            sign_name = CLASS_NAMES.get(pred_class, f"Class {pred_class}")
            category = get_sign_category(pred_class)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🏷️ {sign_name}</h2>
                <p style="font-size: 1.3rem;">Class {pred_class} | {category}</p>
                <p style="font-size: 2rem; font-weight: bold;">{confidence*100:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Safety warning
            if pred_class in SAFETY_CRITICAL_CLASSES:
                st.markdown(f"""
                <div class="safety-critical">
                    <strong>⚠️ Safety-Critical Sign Detected</strong><br>
                    This sign type requires high confidence for autonomous vehicle decisions.
                    {"✅ Confidence is sufficient (>95%)" if confidence > 0.95 else "❌ LOW CONFIDENCE — Manual verification recommended!"}
                </div>
                """, unsafe_allow_html=True)
            
            # Top-5 predictions bar chart
            st.markdown("### 📊 Top-5 Predictions")
            top5_idx = np.argsort(probs)[-5:][::-1]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#667eea' if i == pred_class else '#b0bec5' for i in top5_idx]
            names = [f"[{i}] {CLASS_NAMES.get(i, '?')[:25]}" for i in top5_idx]
            values = [probs[i] * 100 for i in top5_idx]
            
            bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor='white')
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            
            for bar, val in zip(bars, values[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Grad-CAM section
        st.markdown("---")
        st.markdown("### 🔍 Model Attention (Grad-CAM)")
        
        gcol1, gcol2, gcol3 = st.columns(3)
        
        heatmap, _, _ = generate_gradcam(model, input_tensor)
        
        # Original
        img_display = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        
        with gcol1:
            st.image(img_display, caption="Input Image", use_container_width=True)
        
        with gcol2:
            fig_h, ax_h = plt.subplots(figsize=(4, 4))
            ax_h.imshow(heatmap, cmap='jet')
            ax_h.axis('off')
            ax_h.set_title("Attention Heatmap")
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        
        with gcol3:
            img_float = img_display.astype(np.float32) / 255.0
            heatmap_resized = cv2.resize(heatmap, (img_float.shape[1], img_float.shape[0]))
            heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.5 * img_float + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            st.image(overlay, caption="Overlay", use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please upload a valid image file (JPG, PNG, or BMP).")

else:
    # Show instructions when no image is uploaded
    st.markdown("---")
    st.info("👆 Upload a traffic sign image to get started, or use the sample images below.")
    
    # Display class reference
    st.markdown("### 📋 GTSRB Class Reference (43 Categories)")
    
    cols = st.columns(3)
    for i, (cls_id, name) in enumerate(CLASS_NAMES.items()):
        with cols[i % 3]:
            emoji = "🔴" if cls_id in SAFETY_CRITICAL_CLASSES else "🔵"
            st.markdown(f"{emoji} **{cls_id}**: {name}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    Traffic Sign Recognition for Autonomous Vehicles | Project #19<br>
    GTSRB Dataset | CNN with Spatial Transformer Network<br>
    Built with PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
