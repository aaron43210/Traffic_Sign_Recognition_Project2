"""
Evaluation Module
=================
Comprehensive model evaluation with:
- Accuracy, Precision, Recall, F1 (macro, weighted, per-class)
- Confusion matrix visualization (43×43)
- Per-class accuracy analysis
- Top-K accuracy
- ROC curves for safety-critical classes
- Model comparison table
- Inference speed benchmarking
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from collections import defaultdict

from src.config import (
    NUM_CLASSES, CLASS_NAMES, DEVICE, SAFETY_CRITICAL_CLASSES,
    SIGN_CATEGORIES, FIGURES_DIR
)
from src.utils import save_figure, create_plot_style, format_metrics


def get_predictions(model, dataloader, device=DEVICE):
    """
    Get all predictions and ground truth labels from a dataloader.
    
    Returns
    -------
    dict with keys:
        'y_true': np.array of true labels
        'y_pred': np.array of predicted labels
        'y_prob': np.array of prediction probabilities (N x num_classes)
        'images': list of input tensors (optional, if store_images=True)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'y_true': np.array(all_labels),
        'y_pred': np.array(all_preds),
        'y_prob': np.array(all_probs),
    }


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive classification metrics.
    
    Returns
    -------
    dict
        Dictionary of metric name → value
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


def compute_topk_accuracy(y_prob, y_true, k=5):
    """Compute top-K accuracy."""
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = sum(1 for i, label in enumerate(y_true) if label in top_k_preds[i])
    return correct / len(y_true)


def evaluate_model(model, test_loader, model_name="Model", device=DEVICE):
    """
    Full evaluation pipeline for a single model.
    
    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    model_name : str
    device : torch.device
    
    Returns
    -------
    dict
        All metrics and predictions
    """
    model = model.to(device)
    predictions = get_predictions(model, test_loader, device)
    
    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    y_prob = predictions['y_prob']
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics['Top-5 Accuracy'] = compute_topk_accuracy(y_prob, y_true, k=5)
    
    # Print results
    print(format_metrics(metrics, f"{model_name} — Test Results"))
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES.get(i, f"Class {i}")[:30] for i in range(NUM_CLASSES)],
        zero_division=0,
        output_dict=True
    )
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'report': report,
        'model_name': model_name,
    }


def plot_confusion_matrix(y_true, y_pred, model_name="Model", normalize=True):
    """
    Plot 43×43 confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array
    y_pred : array
    model_name : str
    normalize : bool
        If True, normalize rows to show percentages
    """
    create_plot_style()
    
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        plot_data = cm_normalized
        fmt = '.1%'
        title_suffix = "(Normalized)"
    else:
        plot_data = cm
        fmt = 'd'
        title_suffix = "(Counts)"
    
    fig, ax = plt.subplots(figsize=(22, 20))
    
    sns.heatmap(
        plot_data, annot=False, fmt=fmt, cmap='Blues',
        xticklabels=range(NUM_CLASSES),
        yticklabels=range(NUM_CLASSES),
        ax=ax, linewidths=0.1, linecolor='white',
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel("Predicted Class", fontsize=13)
    ax.set_ylabel("True Class", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name} {title_suffix}",
                 fontsize=15, fontweight='bold', pad=15)
    ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_figure(fig, fname)
    return fig


def plot_per_class_accuracy(y_true, y_pred, model_name="Model"):
    """
    Bar chart showing per-class accuracy for all 43 classes.
    Highlights safety-critical classes.
    """
    create_plot_style()
    
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    colors = ['#FF4444' if i in SAFETY_CRITICAL_CLASSES else '#4ECDC4'
              for i in range(NUM_CLASSES)]
    
    bars = ax.bar(range(NUM_CLASSES), per_class_acc * 100, color=colors,
                  edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Class ID", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"Per-Class Accuracy — {model_name}\n(Red = Safety-Critical Classes)",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(range(NUM_CLASSES), fontsize=7)
    ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(per_class_acc) * 100, color='red', linestyle='--',
               alpha=0.5, label=f'Mean: {np.mean(per_class_acc)*100:.1f}%')
    ax.legend(fontsize=11)
    
    # Annotate lowest accuracy classes
    worst_5 = np.argsort(per_class_acc)[:5]
    for idx in worst_5:
        ax.annotate(f"{per_class_acc[idx]*100:.1f}%",
                    xy=(idx, per_class_acc[idx] * 100),
                    xytext=(0, 15), textcoords='offset points',
                    ha='center', fontsize=7, fontweight='bold', color='red')
    
    plt.tight_layout()
    save_figure(fig, f"per_class_accuracy_{model_name.lower().replace(' ', '_')}.png")
    return fig


def plot_roc_curves(y_true, y_prob, model_name="Model"):
    """
    Plot ROC curves for safety-critical classes (one-vs-rest).
    """
    create_plot_style()
    
    # Binarize labels
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    # Select safety-critical classes to plot
    critical_classes = [14, 13, 17, 0, 1, 2, 5, 8, 25, 28]  # Most important
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(critical_classes)))
    
    for i, cls_id in enumerate(critical_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, cls_id], y_prob[:, cls_id])
        roc_auc = auc(fpr, tpr)
        
        short_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")[:25]
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{cls_id}: {short_name} (AUC={roc_auc:.4f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — Safety-Critical Classes\n{model_name}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    save_figure(fig, f"roc_curves_{model_name.lower().replace(' ', '_')}.png")
    return fig


def benchmark_inference_speed(model, input_size=(1, 3, 48, 48),
                               num_iterations=100, device=DEVICE):
    """
    Benchmark model inference speed (FPS).
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    elapsed = time.time() - start
    
    fps = num_iterations / elapsed
    ms_per_image = 1000 * elapsed / num_iterations
    
    result = {
        'device': str(device),
        'fps': fps,
        'ms_per_image': ms_per_image,
        'num_iterations': num_iterations,
    }
    
    print(f"\n[⚡] Inference Benchmark ({device}):")
    print(f"     {fps:.1f} FPS | {ms_per_image:.2f} ms/image")
    
    return result


def compare_models(results_list):
    """
    Create a comparison table and visualization for multiple models.
    
    Parameters
    ----------
    results_list : list of dict
        Each dict from evaluate_model()
    """
    create_plot_style()
    
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)
    
    header = f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>10} {'Top-5 Acc':>10}"
    print(header)
    print("-" * 80)
    
    names = []
    accuracies = []
    f1_scores = []
    
    for result in results_list:
        name = result['model_name']
        m = result['metrics']
        print(f"{name:<20} {m['Accuracy']*100:>9.2f}% {m['Macro F1']*100:>9.2f}% "
              f"{m['Weighted F1']*100:>9.2f}% {m.get('Top-5 Accuracy', 0)*100:>9.2f}%")
        names.append(name)
        accuracies.append(m['Accuracy'] * 100)
        f1_scores.append(m['Macro F1'] * 100)
    
    print("=" * 80)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison", fontsize=15, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(names)]
    
    axes[0].bar(names, accuracies, color=colors, edgecolor='white')
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(90, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
    
    axes[1].bar(names, f1_scores, color=colors, edgecolor='white')
    axes[1].set_ylabel("Macro F1 (%)")
    axes[1].set_title("Macro F1 Score")
    axes[1].set_ylim(90, 100)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.2, f"{v:.2f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "model_comparison.png")
    return fig
