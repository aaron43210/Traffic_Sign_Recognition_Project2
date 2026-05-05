#  Traffic Sign Recognition for Autonomous Vehicles

> **Project #19** | German Traffic Sign Recognition Benchmark (GTSRB)  
> Predictive Analytics Course Project | Academic Year 2025-26

---

## 👥 Team Members
| Name | Role | GitHub | Contribution Profile |
|------|------|--------|----------------------|
| Aaron R | Data Pipeline, Model Architecture & Training | https://github.com/aaron43210 | Stages 1, 2, 5, 6, 7, 9 |
| Ardra Selin A G | Preprocessing, EDA, Safety Analysis & UI | https://github.com/ardraselin22| Stages 3, 4, 6, 8, 10 |

*(Note: Both members collaborated across all stages to meet the project guidelines, with primary responsibilities distributed as above. We split the codebase commits to reflect these contributions.)*

---

## 📌 Problem Statement & Motivation
Autonomous vehicles rely heavily on computer vision to navigate safely. Misinterpreting a traffic sign—such as confusing a "Stop" sign with a "Yield" sign—can lead to catastrophic accidents. 
This project builds a robust deep learning system to classify **43 categories** of traffic signs under varying real-world conditions (lighting, occlusion, motion blur). Our system goes beyond simple classification by prioritizing **safety-critical analysis** (ISO 26262 ASIL) and **model explainability**.

## 📊 Dataset Description
- **Source:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Size:** ~39,209 training images | ~12,630 test images
- **Features:** RGB images, varying resolutions (15×15 to 250×250 pixels), standardized to 48×48.
- **Class Distribution:** Highly imbalanced (ranging from 210 to 2,250 samples per class). We handled this using Weighted Random Sampling and Weighted Cross-Entropy Loss.

## 🔬 Methodology Overview (10 Stages)
We strictly followed the Data Science Life Cycle:
1. **Problem Definition:** Classify 43 sign types for autonomous driving.
2. **Data Collection:** Automated fetching of GTSRB via PyTorch.
3. **Data Preprocessing:** Implemented **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to normalize varying lighting.
4. **EDA:** Analyzed class imbalance, brightness distributions, and visual similarities.
5. **Feature Engineering:** Heavy data augmentation (rotation, perspective, color jitter).
6. **Model Building:** Compared Baseline CNN, Enhanced CNN, and our flagship **STN-CNN** (Spatial Transformer Network).
7. **Model Evaluation:** Analyzed Macro F1, Precision/Recall, and Per-Class metrics.
8. **Explainability:** Used **Grad-CAM** and t-SNE to visualize model attention and feature embeddings.
9. **Deployment:** Interactive **Streamlit** application with confidence gauges and lighting simulation.
10. **Documentation:** Comprehensive README, codebase, and GitHub profiles.

## 📈 Results Summary
| Model | Test Accuracy | Macro F1 | Top-5 Accuracy | Parameters |
|-------|:------------:|:--------:|:--------------:|:----------:|
| Baseline CNN | ~96-97% | ~95% | ~99% | ~500K |
| Enhanced CNN | ~98-99% | ~98% | ~99.5% | ~1.2M |
| **STN-CNN** ⭐ | **~99%+** | **~99%** | **~99.9%** | ~1.5M |

## 🛡️ Safety Analysis
- **Critical Failures:** Identified Stop ↔ Yield and speed limit confusions as high-risk (ASIL C/D).
- **Recommendations:** Multi-sensor fusion, 95% confidence thresholding, and temporal consistency.

## 🖥️ Streamlit Deployment

🔗 **[Live Streamlit Application Link](https://trafficsignrecognitionproject2.streamlit.app/)**

### App Screenshots

![Streamlit App Interface](./figures/streamlit_ui_demo.png)

![Grad-CAM Explainability](./figures/streamlit_gradcam_demo.png)


## 🚀 Setup & Run Local  <img sly

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd PROJECT2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis Notebook
```bash
jupyter notebook project2.ipynb
```

### Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

## 📁 Repository Structure
```
PROJECT2/
├── project2.ipynb              # Main notebook covering all 10 stages
├── src/                        # Modular source code
│   ├── config.py               # Hyperparameters
│   ├── data_loader.py          # Dataset fetching & loaders
│   ├── preprocessing.py        # CLAHE & transformations
│   ├── augmentation.py         # Data augmentation
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── model.py                # STN and CNN architectures
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation & metrics
│   ├── explainability.py       # Grad-CAM and t-SNE
│   ├── failure_analysis.py     # Safety and error analysis
│   ├── lighting_analysis.py    # Robustness testing
│   └── utils.py                # Helpers
├── streamlit_app.py            # Streamlit UI
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── individual_profiles/        # GitHub contribution screenshots
└── figures/                    # Generated charts and diagrams
```

## 📚 References
1. Stallkamp, J., et al. "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition." Neural Networks, 2012.
2. Jaderberg, M., et al. "Spatial Transformer Networks." NeurIPS, 2015.
3. Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV, 2017.

---

