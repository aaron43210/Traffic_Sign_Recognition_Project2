"""
Failure Analysis & Safety Implications Module
================================================
Systematically identifies and analyzes misclassification patterns,
with a focus on safety-critical implications for autonomous vehicles.

Analysis includes:
- Top confusion pairs identification
- Failure case visualization
- Root cause categorization (similar appearance, lighting, occlusion, etc.)
- Safety risk classification (benign vs. dangerous misclassifications)
- ISO 26262 / ASIL-level discussion
- Recommendations for real-world deployment
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

from src.config import (
    NUM_CLASSES, CLASS_NAMES, SAFETY_CRITICAL_CLASSES,
    SIGN_CATEGORIES, DEVICE, FIGURES_DIR, IMG_SIZE,
    DATASET_MEAN, DATASET_STD
)
from src.utils import save_figure, create_plot_style


def analyze_failures(model, test_loader, device=DEVICE):
    """
    Identify all misclassified samples and analyze failure patterns.
    
    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    device : torch.device
    
    Returns
    -------
    dict
        Complete failure analysis results
    """
    model.eval()
    model = model.to(device)
    
    misclassified = []  # List of (image, true_label, pred_label, confidence, prob)
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            
            for i in range(len(labels)):
                total_count += 1
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = confidences[i].item()
                prob_vec = probs[i].cpu().numpy()
                
                if pred_label != true_label:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence,
                        'prob': prob_vec,
                    })
                else:
                    correct_count += 1
    
    # Analyze confusion pairs
    confusion_pairs = Counter()
    for m in misclassified:
        pair = (m['true_label'], m['pred_label'])
        confusion_pairs[pair] += 1
    
    # Classify by safety risk
    safety_failures = []
    benign_failures = []
    
    for m in misclassified:
        if (m['true_label'] in SAFETY_CRITICAL_CLASSES or
            m['pred_label'] in SAFETY_CRITICAL_CLASSES):
            safety_failures.append(m)
        else:
            benign_failures.append(m)
    
    results = {
        'total': total_count,
        'correct': correct_count,
        'misclassified_count': len(misclassified),
        'misclassified': misclassified,
        'confusion_pairs': confusion_pairs,
        'safety_failures': safety_failures,
        'benign_failures': benign_failures,
        'accuracy': correct_count / total_count if total_count > 0 else 0,
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("  FAILURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total test samples:        {total_count}")
    print(f"  Correctly classified:      {correct_count}")
    print(f"  Misclassified:             {len(misclassified)}")
    print(f"  Overall accuracy:          {results['accuracy']*100:.2f}%")
    print(f"  Safety-critical failures:  {len(safety_failures)}")
    print(f"  Benign failures:           {len(benign_failures)}")
    print("=" * 60)
    
    return results


def plot_top_confusion_pairs(failure_results, top_n=15):
    """
    Visualize the most common misclassification pairs.
    """
    create_plot_style()
    
    pairs = failure_results['confusion_pairs'].most_common(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = []
    counts = []
    colors = []
    
    for (true_cls, pred_cls), count in pairs:
        true_name = CLASS_NAMES.get(true_cls, f"C{true_cls}")[:20]
        pred_name = CLASS_NAMES.get(pred_cls, f"C{pred_cls}")[:20]
        labels.append(f"{true_cls}→{pred_cls}\n{true_name}\n→ {pred_name}")
        counts.append(count)
        
        # Red if safety-critical
        if true_cls in SAFETY_CRITICAL_CLASSES or pred_cls in SAFETY_CRITICAL_CLASSES:
            colors.append('#FF4444')
        else:
            colors.append('#4ECDC4')
    
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of Misclassifications", fontsize=12)
    ax.set_title(f"Top {top_n} Confusion Pairs\n(Red = Safety-Critical)",
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "top_confusion_pairs.png")
    return fig


def plot_failure_examples(failure_results, num_examples=20):
    """
    Display grid of misclassified images with true/predicted labels.
    """
    create_plot_style()
    
    misclassified = failure_results['misclassified']
    
    # Sort by confidence (show most confident wrong predictions)
    sorted_failures = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)
    display_failures = sorted_failures[:num_examples]
    
    n_cols = 5
    n_rows = (num_examples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    fig.suptitle("Most Confident Misclassifications (Highest Risk)",
                 fontsize=14, fontweight='bold', y=1.02)
    
    mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
    std = torch.tensor(DATASET_STD).view(3, 1, 1)
    
    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        if i < len(display_failures):
            fail = display_failures[i]
            
            # Denormalize image
            img = fail['image'] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            true_name = CLASS_NAMES.get(fail['true_label'], '?')[:18]
            pred_name = CLASS_NAMES.get(fail['pred_label'], '?')[:18]
            
            is_safety = (fail['true_label'] in SAFETY_CRITICAL_CLASSES or
                         fail['pred_label'] in SAFETY_CRITICAL_CLASSES)
            color = 'red' if is_safety else 'orange'
            
            ax.set_title(f"True: {fail['true_label']} ({true_name})\n"
                         f"Pred: {fail['pred_label']} ({pred_name})\n"
                         f"Conf: {fail['confidence']*100:.1f}%",
                         fontsize=7, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, "failure_examples.png")
    return fig


def generate_safety_report(failure_results):
    """
    Generate a comprehensive safety implications report.
    
    Returns
    -------
    str
        Formatted safety report text
    """
    total = failure_results['total']
    misclassified = failure_results['misclassified']
    safety_failures = failure_results['safety_failures']
    confusion_pairs = failure_results['confusion_pairs']
    
    report = []
    report.append("\n" + "=" * 70)
    report.append("  SAFETY IMPLICATIONS REPORT FOR AUTONOMOUS VEHICLES")
    report.append("=" * 70)
    
    # 1. Overall safety risk
    safety_rate = len(safety_failures) / total * 100 if total > 0 else 0
    report.append(f"\n  1. OVERALL SAFETY RISK")
    report.append(f"     Total test images:           {total}")
    report.append(f"     Total misclassifications:    {len(misclassified)} "
                  f"({len(misclassified)/total*100:.2f}%)")
    report.append(f"     Safety-critical failures:    {len(safety_failures)} "
                  f"({safety_rate:.3f}%)")
    
    # 2. Critical confusion analysis
    report.append(f"\n  2. CRITICAL CONFUSION ANALYSIS")
    report.append(f"     Most dangerous misclassification patterns:")
    
    dangerous_pairs = [
        ((14, 13), "Stop → Yield", "CRITICAL: Vehicle may not stop at stop sign"),
        ((13, 14), "Yield → Stop", "MODERATE: Unnecessary stops, traffic disruption"),
        ((17, 14), "No Entry → Stop", "CRITICAL: Wrong action at restricted area"),
        ((14, 17), "Stop → No Entry", "HIGH: May enter restricted/oncoming traffic"),
    ]
    
    for (true_c, pred_c), name, implication in dangerous_pairs:
        count = confusion_pairs.get((true_c, pred_c), 0)
        if count > 0:
            report.append(f"     • {name}: {count} cases")
            report.append(f"       Implication: {implication}")
    
    # Check speed limit confusions
    speed_classes = [0, 1, 2, 3, 4, 5, 7, 8]
    speed_confusions = 0
    for (true_c, pred_c), count in confusion_pairs.items():
        if true_c in speed_classes and pred_c in speed_classes and true_c != pred_c:
            speed_confusions += count
    
    if speed_confusions > 0:
        report.append(f"\n     Speed limit confusions: {speed_confusions} total")
        report.append(f"       Implication: Vehicle may drive at wrong speed,")
        report.append(f"       risking speeding violations or causing accidents")
    
    # 3. ASIL Risk Classification
    report.append(f"\n  3. ISO 26262 RISK CLASSIFICATION (ASIL)")
    report.append(f"     ┌─────────────────────────────────────────────────────┐")
    report.append(f"     │ ASIL D (Highest): Stop sign, No entry misclass.    │")
    report.append(f"     │ ASIL C:           Speed limit confusions            │")
    report.append(f"     │ ASIL B:           Yield, Priority road misclass.    │")
    report.append(f"     │ ASIL A:           Warning sign confusion            │")
    report.append(f"     │ QM (No safety):   End-of-restriction confusions     │")
    report.append(f"     └─────────────────────────────────────────────────────┘")
    
    # 4. Recommendations
    report.append(f"\n  4. DEPLOYMENT RECOMMENDATIONS")
    report.append(f"     a. MULTI-SENSOR FUSION: Never rely solely on camera-based")
    report.append(f"        sign recognition. Fuse with HD map data and V2X signals.")
    report.append(f"     b. CONFIDENCE THRESHOLDING: Reject predictions with")
    report.append(f"        confidence < 95% and trigger human takeover request.")
    report.append(f"     c. TEMPORAL CONSISTENCY: Require the same prediction")
    report.append(f"        across 3-5 consecutive frames before acting.")
    report.append(f"     d. REDUNDANT MODELS: Deploy ensemble of diverse models")
    report.append(f"        and require consensus for safety-critical signs.")
    report.append(f"     e. ADVERSARIAL ROBUSTNESS: Test against adversarial")
    report.append(f"        patches and perturbations before deployment.")
    report.append(f"     f. CONTINUOUS MONITORING: Implement OOD detection to")
    report.append(f"        flag unfamiliar sign types or conditions.")
    report.append(f"     g. HUMAN-IN-THE-LOOP: Maintain driver override capability")
    report.append(f"        per SAE Level 3+ requirements.")
    
    report.append(f"\n  5. FAILURE MODE CATEGORIZATION")
    
    # Categorize failure modes
    high_conf_wrong = [m for m in misclassified if m['confidence'] > 0.9]
    low_conf_wrong = [m for m in misclassified if m['confidence'] < 0.5]
    
    report.append(f"     High-confidence wrong predictions (>90%): {len(high_conf_wrong)}")
    report.append(f"       → These are the MOST DANGEROUS: model is confidently wrong")
    report.append(f"     Low-confidence wrong predictions (<50%):  {len(low_conf_wrong)}")
    report.append(f"       → These could be caught by confidence thresholding")
    
    catchable = len(low_conf_wrong)
    report.append(f"\n     With 50% confidence threshold:")
    report.append(f"       Catchable failures: {catchable}/{len(misclassified)} "
                  f"({catchable/max(len(misclassified),1)*100:.1f}%)")
    
    report.append("\n" + "=" * 70)
    
    full_report = "\n".join(report)
    print(full_report)
    
    return full_report


def plot_safety_risk_matrix(failure_results):
    """
    Create a risk matrix visualization: severity vs. frequency of failures.
    """
    create_plot_style()
    
    confusion_pairs = failure_results['confusion_pairs']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Categorize each confusion pair by severity and frequency
    for (true_cls, pred_cls), count in confusion_pairs.items():
        # Severity: based on how safety-critical the classes are
        true_critical = true_cls in SAFETY_CRITICAL_CLASSES
        pred_critical = pred_cls in SAFETY_CRITICAL_CLASSES
        
        if true_critical and pred_critical:
            severity = 3  # Critical
            color = '#FF0000'
        elif true_critical or pred_critical:
            severity = 2  # High
            color = '#FF8800'
        else:
            severity = 1  # Low
            color = '#44BB44'
        
        ax.scatter(count, severity + np.random.normal(0, 0.1),
                   s=count * 20, alpha=0.6, color=color, edgecolors='black',
                   linewidth=0.5)
        
        if count >= 3:
            ax.annotate(f"{true_cls}→{pred_cls}",
                        xy=(count, severity), fontsize=7,
                        ha='center', va='bottom')
    
    ax.set_xlabel("Frequency of Misclassification", fontsize=12)
    ax.set_ylabel("Safety Severity", fontsize=12)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Low\n(Non-critical signs)",
                         "High\n(One critical sign)",
                         "Critical\n(Both critical signs)"])
    ax.set_title("Safety Risk Matrix: Severity vs Frequency",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add risk zone shading
    ax.axhspan(2.5, 3.5, alpha=0.1, color='red', label='Critical Risk Zone')
    ax.axhspan(1.5, 2.5, alpha=0.1, color='orange', label='High Risk Zone')
    ax.axhspan(0.5, 1.5, alpha=0.1, color='green', label='Low Risk Zone')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, "safety_risk_matrix.png")
    return fig
