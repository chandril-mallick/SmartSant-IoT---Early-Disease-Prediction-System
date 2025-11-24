"""
Stool Image Model Evaluation
Comprehensive evaluation with all metrics and visualizations for Bristol Stool Scale classification.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config
from models.feature_extraction_demo import StoolFeatureExtractor
from preprocessing.stool_image_preprocessor import StoolImagePreprocessor


def compute_specificity(y_true, y_pred, num_classes=7):
    """
    Compute specificity for multi-class classification.
    
    For multi-class, we compute specificity for each class and average.
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    
    for i in range(num_classes):
        # True Negatives: all correct predictions except class i
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # False Positives: column i sum minus diagonal
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = 0.0
        
        specificities.append(spec)
    
    return np.mean(specificities)


def evaluate_multiclass_model(
    y_true,
    y_pred,
    y_pred_proba=None,
    class_names=None,
    model_name="Stool Classification Model"
):
    """
    Comprehensive evaluation for multi-class classification.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_pred_proba: Predicted probabilities [N, num_classes]
        class_names: List of class names
        model_name: Name for display
        
    Returns:
        Dictionary with all metrics
    """
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(num_classes)]
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    specificity = compute_specificity(y_true, y_pred, num_classes)
    
    # ROC-AUC (one-vs-rest)
    roc_auc = None
    if y_pred_proba is not None:
        try:
            # Binarize labels for one-vs-rest
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            roc_auc = None
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*70}\n")
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:     {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision:    {precision:.4f}  ({precision*100:.2f}%) [Macro Average]")
    print(f"  Recall:       {recall:.4f}  ({recall*100:.2f}%) [Sensitivity, Macro]")
    print(f"  F1-Score:     {f1:.4f}  [Macro Average]")
    print(f"  Specificity:  {specificity:.4f}  ({specificity*100:.2f}%) [Macro Average]")
    if roc_auc is not None:
        print(f"  ROC-AUC:      {roc_auc:.4f}  [Macro Average, OvR]")
    
    print(f"\n{'='*70}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("(Rows: True labels, Columns: Predicted labels)\n")
    
    # Create header
    header = "True \\ Pred |"
    for name in class_names:
        header += f" {name:^8} |"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, true_name in enumerate(class_names):
        row = f" {true_name:^10} |"
        for j in range(num_classes):
            row += f" {cm[i][j]:^8} |"
        print(row)
    print()
    
    # Per-class metrics
    print("Per-Class Metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 65)
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(class_names):
        support = np.sum(y_true == i)
        print(f"{name:<15} {precision_per_class[i]:>10.4f} {recall_per_class[i]:>10.4f} "
              f"{f1_per_class[i]:>10.4f} {support:>10}")
    
    print()
    
    # Package results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'sensitivity': recall,
        'f1_macro': f1,
        'specificity': specificity,
        'roc_auc_macro': roc_auc,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show actual counts
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix\n(Numbers show actual counts, colors show normalized frequencies)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot ROC curves for multi-class classification (one-vs-rest)."""
    num_classes = len(class_names)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves (One-vs-Rest)\nBristol Stool Scale Classification')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {save_path}")
    
    plt.close()


def plot_precision_recall_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot precision-recall curves for multi-class classification."""
    num_classes = len(class_names)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i],
            y_pred_proba[:, i]
        )
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            recall[i], precision[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves\nBristol Stool Scale Classification')
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Precision-recall curves saved to: {save_path}")
    
    plt.close()


def evaluate_stool_model(
    model=None,
    test_loader=None,
    device='cpu',
    save_dir=None
):
    """
    Complete evaluation pipeline for stool image model.
    
    Args:
        model: Trained model (if None, creates and uses random predictions)
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all results
    """
    if save_dir is None:
        save_dir = os.path.join(data_config.MODELS_DIR, 'stool_evaluation')
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = [f"Type {i+1}" for i in range(7)]
    
    # If no model provided, create one for demonstration
    if model is None:
        print("\n⚠️  No trained model provided. Creating model for demonstration...")
        model = StoolFeatureExtractor(
            model_name='efficientnet_b0',
            num_classes=7,
            pretrained=True,
            freeze_backbone=True
        )
        model.eval()
    
    # Collect predictions
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\n🔄 Making predictions on test set...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            
            # Get predictions
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    print(f"✅ Predictions complete: {len(all_labels)} samples")
    
    # Evaluate
    results = evaluate_multiclass_model(
        y_true=all_labels,
        y_pred=all_preds,
        y_pred_proba=all_probs,
        class_names=class_names,
        model_name="Bristol Stool Scale CNN"
    )
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # ROC curves
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plot_roc_curves(all_labels, all_probs, class_names, roc_path)
    
    # Precision-recall curves
    pr_path = os.path.join(save_dir, 'precision_recall_curves.png')
    plot_precision_recall_curves(all_labels, all_probs, class_names, pr_path)
    
    # Save results
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    results_to_save = {k: v for k, v in results.items() if k != 'confusion_matrix'}
    results_to_save['confusion_matrix'] = results['confusion_matrix'].tolist()
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")
    
    # Save text summary
    summary_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Bristol Stool Scale Classification - Evaluation Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy:     {results['accuracy']:.4f}\n")
        f.write(f"Precision:    {results['precision_macro']:.4f}\n")
        f.write(f"Recall:       {results['recall_macro']:.4f}\n")
        f.write(f"F1-Score:     {results['f1_macro']:.4f}\n")
        f.write(f"Specificity:  {results['specificity']:.4f}\n")
        if results['roc_auc_macro']:
            f.write(f"ROC-AUC:      {results['roc_auc_macro']:.4f}\n")
    
    print(f"✅ Summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STOOL IMAGE MODEL EVALUATION")
    print("="*70)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    preprocessor = StoolImagePreprocessor(
        image_size=224,
        train_val_test_split=(0.7, 0.15, 0.15)
    )
    
    _, _, test_loader = preprocessor.process_all(
        filter_quality=False,
        batch_size=8
    )
    
    # Run evaluation
    results = evaluate_stool_model(
        model=None,  # Will create a model with random weights for demo
        test_loader=test_loader,
        device='cpu'
    )
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: models/stool_evaluation/")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curves.png")
    print(f"   - precision_recall_curves.png")
    print(f"   - evaluation_results.json")
    print(f"   - evaluation_summary.txt")
