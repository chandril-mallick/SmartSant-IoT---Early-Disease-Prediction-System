"""
Comprehensive Model Evaluation
Computes all metrics and creates visualizations for optimized urine classifier.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.urine_preprocessor import preprocess_urine_data
from config import data_config


def compute_specificity(y_true, y_pred):
    """
    Compute specificity (True Negative Rate).
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity


def evaluate_model_complete(model, X_test, y_test, threshold=0.5, model_name="Model"):
    """
    Comprehensive evaluation with all requested metrics.
    
    Returns:
        Dictionary with all metrics
    """
    print("\n" + "="*70)
    print(f"{model_name} - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = compute_specificity(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n📊 Test Set Performance:")
    print(f"  {'Metric':<20} {'Value':<10} {'Percentage'}")
    print("  " + "-"*50)
    print(f"  {'Accuracy':<20} {accuracy:<10.4f} {accuracy*100:>6.2f}%")
    print(f"  {'Precision':<20} {precision:<10.4f} {precision*100:>6.2f}%")
    print(f"  {'Recall (Sensitivity)':<20} {recall:<10.4f} {recall*100:>6.2f}%")
    print(f"  {'F1-Score':<20} {f1:<10.4f} {f1*100:>6.2f}%")
    print(f"  {'Specificity':<20} {specificity:<10.4f} {specificity*100:>6.2f}%")
    print(f"  {'ROC-AUC':<20} {roc_auc:<10.4f} {roc_auc*100:>6.2f}%")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    
    print(f"\n🎯 Clinical Interpretation:")
    print(f"  Correctly identified healthy: {tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)")
    print(f"  Correctly identified UTI:     {tp}/{tp+fn} ({tp/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%)")
    print(f"  False alarms (FP):            {fp}/{fp+tp} ({fp/(fp+tp)*100 if (fp+tp)>0 else 0:.1f}%)")
    print(f"  Missed cases (FN):            {fn}/{tp+fn} ({fn/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    
    # Labels
    labels = np.array([['TN', 'FP'], ['FN', 'TP']])
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative', 'Actual Positive'],
        cbar_kws={'label': 'Count'}
    )
    
    # Add labels to cells
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, labels[i, j],
                    ha='center', va='center', fontsize=12, color='red', weight='bold')
    
    plt.title('Confusion Matrix - Optimized Urine Classifier')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix: {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_proba, roc_auc, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall/Sensitivity)')
    plt.title('ROC Curve - Optimized Urine Classifier')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curve: {save_path}")
    
    plt.close()


def create_metrics_summary(results, save_path=None):
    """Create visual summary of all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Optimized Urine Classifier - Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = [
        ('Accuracy', results['accuracy']),
        ('Precision', results['precision']),
        ('Recall\n(Sensitivity)', results['recall']),
        ('F1-Score', results['f1_score']),
        ('Specificity', results['specificity']),
        ('ROC-AUC', results['roc_auc'])
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (ax, (name, value)) in enumerate(zip(axes.flat, metrics)):
        ax.bar([name], [value], color=colors[idx], alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% baseline')
        
        # Add value text
        ax.text(0, value + 0.05, f'{value:.4f}\n({value*100:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved metrics summary: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OPTIMIZED URINE CLASSIFIER - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Load optimized model
    model_path = os.path.join(data_config.MODELS_DIR, 'urine_classifiers', 'optimized_urine_classifier.pkl')
    metadata_path = os.path.join(data_config.MODELS_DIR, 'urine_classifiers', 'optimized_model_metadata.json')
    
    print(f"\n📂 Loading optimized model...")
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata['threshold']
    print(f"✅ Model loaded: {metadata['model_name']}")
    print(f"✅ Optimal threshold: {threshold:.3f}")
    
    # Load test data
    print(f"\n📊 Loading and preprocessing test data...")
    data_path = os.path.join(data_config.RAW_DATA_DIR, data_config.URINE_CSV)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
        data_path=data_path,
        target='Diagnosis',
        test_size=0.2,
        random_state=42,
        handle_imbalance=True
    )
    
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Evaluate
    results = evaluate_model_complete(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name="Optimized Random Forest"
    )
    
    # Create output directory
    eval_dir = os.path.join(data_config.MODELS_DIR, 'urine_evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save visualizations
    print(f"\n📊 Creating visualizations...")
    
    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], cm_path)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_path = os.path.join(eval_dir, 'roc_curve.png')
    plot_roc_curve(y_test, y_proba, results['roc_auc'], roc_path)
    
    summary_path = os.path.join(eval_dir, 'metrics_summary.png')
    create_metrics_summary(results, summary_path)
    
    # Save results
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    results_to_save = {k: v for k, v in results.items() if k != 'confusion_matrix'}
    results_to_save['confusion_matrix'] = results['confusion_matrix'].tolist()
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✅ Saved results: {results_path}")
    
    # Save text summary
    summary_text_path = os.path.join(eval_dir, 'evaluation_summary.txt')
    with open(summary_text_path, 'w') as f:
        f.write("Optimized Urine Classifier - Evaluation Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy:             {results['accuracy']:.4f}\n")
        f.write(f"Precision:            {results['precision']:.4f}\n")
        f.write(f"Recall (Sensitivity): {results['recall']:.4f}\n")
        f.write(f"F1-Score:             {results['f1_score']:.4f}\n")
        f.write(f"Specificity:          {results['specificity']:.4f}\n")
        f.write(f"ROC-AUC:              {results['roc_auc']:.4f}\n")
    
    print(f"✅ Saved summary: {summary_text_path}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All results saved to: {eval_dir}")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curve.png")
    print(f"   - metrics_summary.png")
    print(f"   - evaluation_results.json")
    print(f"   - evaluation_summary.txt")
