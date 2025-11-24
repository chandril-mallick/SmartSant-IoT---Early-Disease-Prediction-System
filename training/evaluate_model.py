"""
Evaluation script for Urine Model
Computes comprehensive evaluation metrics including:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Specificity
- ROC-AUC
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config
from preprocessing.urine_preprocessor import preprocess_urine_data


def compute_specificity(y_true, y_pred):
    """
    Compute specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return specificity


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Comprehensive evaluation of classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = compute_specificity(y_true, y_pred)
    
    # ROC-AUC (requires probabilities)
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    else:
        roc_auc = None
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*60}\n")
    
    print(f"Accuracy:     {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"Precision:    {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f}  ({recall*100:.2f}%) [Sensitivity]")
    print(f"F1-Score:     {f1:.4f}")
    print(f"Specificity:  {specificity:.4f}  ({specificity*100:.2f}%)")
    if roc_auc is not None:
        print(f"ROC-AUC:      {roc_auc:.4f}")
    
    print(f"\n{'='*60}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {cm[0][0]:<10}        {cm[0][1]:<10}")
    print(f"Actual Positive        {cm[1][0]:<10}        {cm[1][1]:<10}")
    print()
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    # Package results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # Same as recall
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return results


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curve saved to: {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def evaluate_with_preprocessor(
    data_path=None,
    target='Diagnosis',
    threshold=0.5,
    save_plots=True
):
    """
    Evaluate model using the preprocessing pipeline.
    Since we don't have a trained neural network model yet,
    we'll use a simple baseline prediction based on the preprocessed data.
    
    Args:
        data_path: Path to the raw data CSV
        target: Target variable name
        threshold: Classification threshold for probabilities
        save_plots: Whether to save evaluation plots
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Set default data path
    if data_path is None:
        data_path = os.path.join(data_config.RAW_DATA_DIR, data_config.URINE_CSV)
    
    print(f"\nLoading and preprocessing data from: {data_path}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
        data_path=data_path,
        target=target,
        test_size=0.2,
        random_state=42,
        handle_imbalance=True,
        sampling_strategy='auto'
    )
    
    print(f"\nTest set size: {len(y_test)}")
    print(f"Test set positive class: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # BASELINE MODEL: Logistic Regression
    # (This is a simple baseline since we don't have a trained neural network yet)
    from sklearn.linear_model import LogisticRegression
    
    print("\nTraining baseline Logistic Regression model...")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = baseline_model.predict(X_test)
    y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        model_name="Baseline Logistic Regression"
    )
    
    # Save plots
    if save_plots:
        plots_dir = os.path.join(data_config.MODELS_DIR, 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # ROC curve
        roc_path = os.path.join(plots_dir, 'roc_curve.png')
        plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
        
        # Confusion matrix
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
    
    return results


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_with_preprocessor(
        target='Diagnosis',
        save_plots=True
    )
    
    # Save results to JSON
    results_path = os.path.join(data_config.MODELS_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Accuracy:     {results['accuracy']:.4f}\n")
        f.write(f"Precision:    {results['precision']:.4f}\n")
        f.write(f"Recall:       {results['recall']:.4f}\n")
        f.write(f"F1-Score:     {results['f1_score']:.4f}\n")
        f.write(f"Specificity:  {results['specificity']:.4f}\n")
        f.write(f"ROC-AUC:      {results['roc_auc']:.4f}\n")
    
    print(f"\n\nResults saved to: {results_path}")
