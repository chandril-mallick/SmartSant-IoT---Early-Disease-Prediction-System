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
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, log_loss
)
from sklearn.preprocessing import label_binarize
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config
from models.feature_extraction_demo import StoolFeatureExtractor
from preprocessing.stool_image_preprocessor import StoolImagePreprocessor


def compute_specificity_from_cm(cm: np.ndarray):
    """
    Compute per-class and macro specificity from a confusion matrix.
    
    Specificity_i = TN_i / (TN_i + FP_i)
    where class i is considered "positive" and all others "negative".
    """
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        # TP, FP, FN, TN for class i (one-vs-rest)
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = 0.0
        specificities.append(spec)

    specificities = np.array(specificities)
    return specificities, float(specificities.mean())


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
        class_names: List of class names (len = num_classes)
        model_name: Name for display
        
    Returns:
        Dictionary with all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Infer number of classes
    if y_pred_proba is not None:
        num_classes = y_pred_proba.shape[1]
    else:
        num_classes = len(np.unique(y_true))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # ensure length consistency
        if len(class_names) != num_classes:
            raise ValueError(
                f"len(class_names)={len(class_names)} != num_classes={num_classes}"
            )

    # Use fixed label order 0..num_classes-1 (important when some classes are missing in y_true)
    labels = list(range(num_classes))

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    precision_micro = precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Confusion matrix & specificity
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_per_class, specificity_macro = compute_specificity_from_cm(cm)

    # Agreement metrics
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC & calibration metrics
    roc_auc_macro = None
    roc_auc_weighted = None
    logloss = None
    brier_score = None

    if y_pred_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=labels)
            roc_auc_macro = roc_auc_score(y_true_bin, y_pred_proba, average="macro", multi_class="ovr")
            roc_auc_weighted = roc_auc_score(y_true_bin, y_pred_proba, average="weighted", multi_class="ovr")
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

        try:
            logloss = log_loss(y_true, y_pred_proba, labels=labels)
        except Exception as e:
            print(f"Warning: Could not compute log-loss: {e}")

        try:
            # Multi-class Brier score (mean squared error between true one-hot and probabilities)
            y_true_bin = label_binarize(y_true, classes=labels)
            brier_score = float(np.mean((y_true_bin - y_pred_proba) ** 2))
        except Exception as e:
            print(f"Warning: Could not compute Brier score: {e}")

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    support_per_class = np.array([(y_true == i).sum() for i in labels])

    # Classification report string
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    # ────────────────────── PRINT SUMMARY ──────────────────────
    print(f"\n{'=' * 70}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'=' * 70}\n")

    print("Overall Metrics (Accuracy / F1 / Balanced):")
    print(f"  Accuracy:                 {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Balanced Accuracy:        {balanced_acc:.4f}  ({balanced_acc*100:.2f}%)")
    print()
    print("Macro Averages (treat all classes equally):")
    print(f"  Precision (macro):        {precision_macro:.4f}")
    print(f"  Recall / Sensitivity:     {recall_macro:.4f}")
    print(f"  F1-score (macro):         {f1_macro:.4f}")
    print(f"  Specificity (macro):      {specificity_macro:.4f}")
    print()
    print("Micro / Weighted Averages:")
    print(f"  Precision (micro):        {precision_micro:.4f}")
    print(f"  Recall (micro):           {recall_micro:.4f}")
    print(f"  F1-score (micro):         {f1_micro:.4f}")
    print(f"  Precision (weighted):     {precision_weighted:.4f}")
    print(f"  Recall (weighted):        {recall_weighted:.4f}")
    print(f"  F1-score (weighted):      {f1_weighted:.4f}")
    print()
    print("Agreement Metrics:")
    print(f"  Cohen's Kappa:            {kappa:.4f}")
    print(f"  Matthews Corr. Coef:      {mcc:.4f}")

    if roc_auc_macro is not None:
        print()
        print("Probability / Calibration Metrics:")
        print(f"  ROC-AUC (macro OvR):      {roc_auc_macro:.4f}")
        if roc_auc_weighted is not None:
            print(f"  ROC-AUC (weighted OvR):   {roc_auc_weighted:.4f}")
        if logloss is not None:
            print(f"  Log-loss:                 {logloss:.4f}")
        if brier_score is not None:
            print(f"  Brier score (multi-class):{brier_score:.4f}")

    print(f"\n{'=' * 70}\n")

    # Pretty confusion matrix
    print("Confusion Matrix:")
    print("(Rows: True labels, Columns: Predicted labels)\n")

    header = "True \\ Pred |"
    for name in class_names:
        header += f" {name:^8} |"
    print(header)
    print("-" * len(header))

    for i, true_name in enumerate(class_names):
        row = f" {true_name:^10} |"
        for j in range(num_classes):
            row += f" {cm[i][j]:^8} |"
        print(row)
    print()

    # Per-class metrics table
    print("Per-Class Metrics:")
    print(f"{'Class':<12} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'F1':>8} {'Support':>10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(
            f"{name:<12} "
            f"{precision_per_class[i]:>8.4f} "
            f"{recall_per_class[i]:>8.4f} "
            f"{specificity_per_class[i]:>8.4f} "
            f"{f1_per_class[i]:>8.4f} "
            f"{support_per_class[i]:>10}"
        )

    print("\nDetailed Classification Report:\n")
    print(clf_report)

    # Pack results
    results = {
        # overall
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        # macro
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "specificity_macro": specificity_macro,
        # micro / weighted
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        # agreement
        "cohen_kappa": kappa,
        "mcc": mcc,
        # auc / calibration
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "log_loss": logloss,
        "brier_score": brier_score,
        # structures
        "confusion_matrix": cm,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "specificity_per_class": specificity_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support_per_class.tolist(),
        "classification_report": clf_report,
        "class_names": class_names,
    }

    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix by row (true class)
    with np.errstate(all="ignore"):
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=cm,  # show actual counts
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Frequency"},
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(
        "Confusion Matrix\n"
        "(Numbers: counts, Colors: row-normalized frequencies)"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Confusion matrix saved to: {save_path}")

    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot ROC curves for multi-class classification (one-vs-rest)."""
    num_classes = len(class_names)
    labels = list(range(num_classes))

    # Binarize labels – important to use all classes in correct order
    y_true_bin = label_binarize(y_true, classes=labels)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))

    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves (One-vs-Rest)\nBristol Stool Scale Classification")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ ROC curves saved to: {save_path}")

    plt.close()


def plot_precision_recall_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot precision-recall curves for multi-class classification."""
    num_classes = len(class_names)
    labels = list(range(num_classes))
    y_true_bin = label_binarize(y_true, classes=labels)

    precision = {}
    recall = {}
    avg_precision = {}

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        avg_precision[i] = average_precision_score(
            y_true_bin[:, i], y_pred_proba[:, i]
        )

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))

    for i, color in zip(range(num_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"{class_names[i]} (AP = {avg_precision[i]:.3f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves\nBristol Stool Scale Classification")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Precision-recall curves saved to: {save_path}")

    plt.close()


def evaluate_stool_model(
    model=None,
    test_loader=None,
    device="cpu",
    save_dir=None
):
    """
    Complete evaluation pipeline for stool image model.
    """
    if save_dir is None:
        save_dir = os.path.join(data_config.MODELS_DIR, "stool_evaluation")
    os.makedirs(save_dir, exist_ok=True)

    class_names = [f"Type {i+1}" for i in range(7)]

    # If no model provided, create one for demonstration
    if model is None:
        print("\n⚠️  No trained model provided. Creating model for demonstration...")
        model = StoolFeatureExtractor(
            model_name="efficientnet_b0",
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=True,
        )
        model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print("\n🔄 Making predictions on test set...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)

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
        model_name="Bristol Stool Scale CNN",
    )

    # Visualizations
    print("\n📊 Creating visualizations...")

    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], class_names, cm_path)

    roc_path = os.path.join(save_dir, "roc_curves.png")
    plot_roc_curves(all_labels, all_probs, class_names, roc_path)

    pr_path = os.path.join(save_dir, "precision_recall_curves.png")
    plot_precision_recall_curves(all_labels, all_probs, class_names, pr_path)

    # Save results JSON
    results_path = os.path.join(save_dir, "evaluation_results.json")
    results_to_save = results.copy()
    # convert numpy arrays / matrix to lists for JSON
    results_to_save["confusion_matrix"] = results["confusion_matrix"].tolist()
    results_to_save["precision_per_class"] = results["precision_per_class"]
    results_to_save["recall_per_class"] = results["recall_per_class"]
    results_to_save["specificity_per_class"] = results["specificity_per_class"]
    results_to_save["f1_per_class"] = results["f1_per_class"]
    results_to_save["support_per_class"] = results["support_per_class"]

    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✅ Results saved to: {results_path}")

    # Save text summary (short)
    summary_path = os.path.join(save_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Bristol Stool Scale Classification - Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy:            {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:   {results['balanced_accuracy']:.4f}\n")
        f.write(f"Precision (macro):   {results['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):      {results['recall_macro']:.4f}\n")
        f.write(f"F1-Score (macro):    {results['f1_macro']:.4f}\n")
        f.write(f"Specificity (macro): {results['specificity_macro']:.4f}\n")
        if results["roc_auc_macro"] is not None:
            f.write(f"ROC-AUC (macro):     {results['roc_auc_macro']:.4f}\n")
        if results["log_loss"] is not None:
            f.write(f"Log-loss:            {results['log_loss']:.4f}\n")
        if results["brier_score"] is not None:
            f.write(f"Brier score:         {results['brier_score']:.4f}\n")

    print(f"✅ Summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STOOL IMAGE MODEL EVALUATION")
    print("=" * 70)

    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    preprocessor = StoolImagePreprocessor(
        image_size=224,
        train_val_test_split=(0.7, 0.15, 0.15),
    )

    _, _, test_loader = preprocessor.process_all(
        filter_quality=False,
        batch_size=8,
    )

    # Run evaluation
    results = evaluate_stool_model(
        model=None,  # Will create a model with random weights for demo
        test_loader=test_loader,
        device="cpu",
    )

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {os.path.join(data_config.MODELS_DIR, 'stool_evaluation')}/")
    print("   - confusion_matrix.png")
    print("   - roc_curves.png")
    print("   - precision_recall_curves.png")
    print("   - evaluation_results.json")
    print("   - evaluation_summary.txt")
