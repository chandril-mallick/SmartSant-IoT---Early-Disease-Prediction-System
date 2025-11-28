"""
Unified System Evaluation Script
Evaluates the performance of the Unified Medical Predictor by testing its components:
1. Urine Disease Classifier (UTI)
2. Kidney Disease Classifier (CKD)

Generates comprehensive metrics and visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config
from inference.unified_medical_predictor import UnifiedMedicalPredictor
from preprocessing.urine_preprocessor import preprocess_urine_data
from preprocessing.kidney_preprocessor import preprocess_kidney_data

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

def evaluate_urine_component(predictor, save_dir):
    """Evaluate the Urine Classifier component."""
    print("\n" + "-"*50)
    print("EVALUATING URINE CLASSIFIER COMPONENT")
    print("-" * 50)
    
    # Load test data
    data_path = os.path.join(data_config.RAW_DATA_DIR, 'urine_data.csv')
    if not os.path.exists(data_path):
        print(f"⚠️  Urine dataset not found at {data_path}. Skipping.")
        return None

    X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
        data_path=data_path,
        target='Diagnosis',
        test_size=0.2,
        random_state=42
    )
    
    # Get model and predictions
    model = predictor.urine_model
    if model is None:
        print("❌ Urine model not loaded in predictor.")
        return None
        
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': auc(*roc_curve(y_test, y_proba)[:2])
    }
    
    print(f"✅ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall:    {metrics['recall']:.4f}")
    print(f"✅ F1-Score:  {metrics['f1']:.4f}")
    print(f"✅ AUC-ROC:   {metrics['auc']:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Urine Classifier Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'urine_confusion_matrix.png'))
    plt.close()
    
    return metrics, y_test, y_pred

def evaluate_kidney_component(predictor, save_dir):
    """Evaluate the Kidney Classifier component."""
    print("\n" + "-"*50)
    print("EVALUATING KIDNEY CLASSIFIER COMPONENT")
    print("-" * 50)
    
    # Load test data
    data_path = os.path.join(data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    if not os.path.exists(data_path):
        print("⚠️  Kidney dataset not found. Skipping.")
        return None, None, None

    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        test_size=0.2,
        random_state=42,
        handle_imbalance=False # Don't SMOTE test data evaluation
    )
    
    # Encode labels
    le = predictor.kidney_label_encoder
    if le is None:
        print("❌ Kidney label encoder not loaded.")
        return None, None, None
        
    y_test_encoded = le.transform(y_test)
    
    # Get model and predictions
    model = predictor.kidney_model
    if model is None:
        print("❌ Kidney model not loaded in predictor.")
        return None, None, None
        
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics (Macro average for multi-class)
    metrics = {
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        'precision': precision_score(y_test_encoded, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test_encoded, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    }
    
    # Calculate AUC (One-vs-Rest)
    try:
        y_test_bin = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
        metrics['auc'] = float(np.mean([
            auc(*roc_curve(y_test_bin[:, i], y_proba[:, i])[:2]) 
            for i in range(len(le.classes_))
        ]))
    except:
        metrics['auc'] = 0.0
    
    print(f"✅ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall:    {metrics['recall']:.4f}")
    print(f"✅ F1-Score:  {metrics['f1']:.4f}")
    print(f"✅ AUC-ROC:   {metrics['auc']:.4f}")
    
    return metrics, y_test_encoded, y_pred, le.classes_

def plot_combined_matrices(urine_data, kidney_data, save_dir):
    """Plot both confusion matrices side-by-side."""
    if not urine_data or not kidney_data:
        print("⚠️  Missing data for combined plot.")
        return

    y_test_u, y_pred_u = urine_data
    y_test_k, y_pred_k, k_classes = kidney_data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Urine Plot
    cm_u = confusion_matrix(y_test_u, y_pred_u)
    sns.heatmap(cm_u, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_title('Urine Classifier (UTI) Confusion Matrix', fontsize=14)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_xticklabels(['Negative', 'Positive'])
    ax1.set_yticklabels(['Negative', 'Positive'])
    
    # Kidney Plot
    cm_k = confusion_matrix(y_test_k, y_pred_k)
    sns.heatmap(cm_k, annot=True, fmt='d', cmap='Greens', cbar=False, 
                xticklabels=k_classes, yticklabels=k_classes, ax=ax2)
    ax2.set_title('Kidney Classifier (CKD) Confusion Matrix', fontsize=14)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unified_confusion_matrices.png'), dpi=300)
    plt.close()
    print(f"✅ Saved combined confusion matrices to: {os.path.join(save_dir, 'unified_confusion_matrices.png')}")

def main():
    print("\n" + "="*70)
    print("UNIFIED SYSTEM EVALUATION")
    print("="*70)
    
    save_dir = os.path.join(data_config.MODELS_DIR, 'unified_evaluation')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Unified Predictor
    predictor = UnifiedMedicalPredictor()
    
    # Evaluate Components
    urine_metrics, y_test_u, y_pred_u = evaluate_urine_component(predictor, save_dir)
    kidney_metrics, y_test_k, y_pred_k, k_classes = evaluate_kidney_component(predictor, save_dir)
    
    # Plot Combined Matrices
    if y_test_u is not None and y_test_k is not None:
        plot_combined_matrices((y_test_u, y_pred_u), (y_test_k, y_pred_k, k_classes), save_dir)
    
    # Save Combined Results
    results = {
        'urine_system': urine_metrics,
        'kidney_system': kidney_metrics,
        'overall_system_health': 'Operational'
    }
    
    with open(os.path.join(save_dir, 'unified_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print(f"✅ Evaluation Complete! Results saved to: {save_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
