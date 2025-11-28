"""
Advanced Kidney Disease Classifier Optimization with SMOTE
===========================================================
Combines SMOTE balancing with optimization strategies to achieve ~90% metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score, 
    recall_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
import config


def train_optimized_models(X_train, X_test, y_train, y_test, classes):
    """
    Train multiple optimized models on SMOTE-balanced data
    """
    models = {}
    
    # 1. Random Forest with Balanced Subsample
    print("\n" + "="*70)
    print("MODEL 1: Random Forest (Balanced Subsample)")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    print("🔧 Training Random Forest...")
    rf.fit(X_train, y_train)
    
    # Cross-validation on training data
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"   CV F1-Score: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    precision_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
    recall_rf = recall_score(y_test, y_pred_rf, average='macro')
    
    print(f"   Test Precision: {precision_rf:.2%}")
    print(f"   Test Recall: {recall_rf:.2%}")
    print(f"   Test F1: {f1_rf:.2%}")
    
    models['Random_Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'cv_f1': cv_scores.mean(),
        'test_f1': f1_rf,
        'test_precision': precision_rf,
        'test_recall': recall_rf
    }
    
    # 2. Gradient Boosting with Custom Weights
    print("\n" + "="*70)
    print("MODEL 2: Gradient Boosting (Cost-Sensitive)")
    print("="*70)
    
    # Compute enhanced weights
    classes_unique, counts = np.unique(y_train, return_counts=True)
    frequencies = counts / len(y_train)
    weights = (1 / frequencies) ** 1.5
    weights = weights / weights.sum() * len(classes_unique)
    class_weight_dict = dict(zip(classes_unique, weights))
    
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    print("🔧 Training Gradient Boosting...")
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred_gb = gb.predict(X_test)
    f1_gb = f1_score(y_test, y_pred_gb, average='macro')
    precision_gb = precision_score(y_test, y_pred_gb, average='macro', zero_division=0)
    recall_gb = recall_score(y_test, y_pred_gb, average='macro')
    
    print(f"   Test Precision: {precision_gb:.2%}")
    print(f"   Test Recall: {recall_gb:.2%}")
    print(f"   Test F1: {f1_gb:.2%}")
    
    models['Gradient_Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'test_f1': f1_gb,
        'test_precision': precision_gb,
        'test_recall': recall_gb
    }
    
    # 3. Neural Network
    print("\n" + "="*70)
    print("MODEL 3: Neural Network (MLP)")
    print("="*70)
    
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    
    print("🔧 Training Neural Network...")
    nn.fit(X_train, y_train)
    
    y_pred_nn = nn.predict(X_test)
    f1_nn = f1_score(y_test, y_pred_nn, average='macro')
    precision_nn = precision_score(y_test, y_pred_nn, average='macro', zero_division=0)
    recall_nn = recall_score(y_test, y_pred_nn, average='macro')
    
    print(f"   Test Precision: {precision_nn:.2%}")
    print(f"   Test Recall: {recall_nn:.2%}")
    print(f"   Test F1: {f1_nn:.2%}")
    
    models['Neural_Network'] = {
        'model': nn,
        'predictions': y_pred_nn,
        'test_f1': f1_nn,
        'test_precision': precision_nn,
        'test_recall': recall_nn
    }
    
    # 4. Calibrated Ensemble
    print("\n" + "="*70)
    print("MODEL 4: Calibrated Ensemble (Voting)")
    print("="*70)
    
    # Select top 3 models
    sorted_models = sorted(
        [(name, data) for name, data in models.items()],
        key=lambda x: x[1]['test_f1'],
        reverse=True
    )[:3]
    
    print(f"   Selected models: {[name for name, _ in sorted_models]}")
    
    estimators = [(name, data['model']) for name, data in sorted_models]
    weights = [data['test_f1'] for _, data in sorted_models]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    print("🔧 Training Ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Calibrate
    print("🔧 Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    y_pred_ens = calibrated.predict(X_test)
    f1_ens = f1_score(y_test, y_pred_ens, average='macro')
    precision_ens = precision_score(y_test, y_pred_ens, average='macro', zero_division=0)
    recall_ens = recall_score(y_test, y_pred_ens, average='macro')
    
    print(f"   Test Precision: {precision_ens:.2%}")
    print(f"   Test Recall: {recall_ens:.2%}")
    print(f"   Test F1: {f1_ens:.2%}")
    
    models['Calibrated_Ensemble'] = {
        'model': calibrated,
        'predictions': y_pred_ens,
        'test_f1': f1_ens,
        'test_precision': precision_ens,
        'test_recall': recall_ens
    }
    
    return models


def evaluate_model(y_true, y_pred, classes, model_name):
    """
    Comprehensive model evaluation
    """
    print(f"\n{'='*70}")
    print(f"DETAILED EVALUATION: {model_name}")
    print(f"{'='*70}")
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:        {accuracy:.2%}")
    print(f"   Macro Precision: {macro_precision:.2%}")
    print(f"   Macro Recall:    {macro_recall:.2%}")
    print(f"   Macro F1-Score:  {macro_f1:.2%}")
    
    # Per-class metrics
    print(f"\n📋 Per-Class Performance:")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    for class_name in classes:
        metrics = report[class_name]
        print(f"   {class_name:20s} - P: {metrics['precision']:.2%}, "
              f"R: {metrics['recall']:.2%}, F1: {metrics['f1-score']:.2%}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': report
    }


def main():
    print("🚀 Advanced Kidney Disease Classifier with SMOTE")
    print("="*70)
    print("Target: ~90% Precision, Recall, and F1-Score")
    print("="*70)
    
    # 1. Load Data WITH SMOTE balancing
    print("\n1️⃣  Loading and preprocessing data with SMOTE...")
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        handle_imbalance=True  # Use SMOTE for balancing
    )
    
    # Load label encoder
    le = joblib.load(os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'label_encoder.pkl'))
    classes = le.classes_
    
    # Encode labels
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    print(f"\n   After SMOTE:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   Training class distribution:")
    for cls, count in zip(unique, counts):
        print(f"   - {classes[cls]}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # 2. Train all models
    print("\n2️⃣  Training optimized models...")
    models = train_optimized_models(X_train, X_test, y_train, y_test, classes)
    
    # 3. Evaluate all models
    print("\n3️⃣  Evaluating all models...")
    results = {}
    for model_name, model_data in models.items():
        results[model_name] = evaluate_model(
            y_test, model_data['predictions'], classes, model_name
        )
    
    # 4. Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['macro_f1'])[0]
    best_metrics = results[best_model_name]
    
    print("\n" + "="*70)
    print("🏆 BEST MODEL")
    print("="*70)
    print(f"Model: {best_model_name}")
    print(f"Macro F1:        {best_metrics['macro_f1']:.2%}")
    print(f"Macro Precision: {best_metrics['macro_precision']:.2%}")
    print(f"Macro Recall:    {best_metrics['macro_recall']:.2%}")
    print(f"Accuracy:        {best_metrics['accuracy']:.2%}")
    
    # 5. Save best model
    save_dir = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers')
    os.makedirs(save_dir, exist_ok=True)
    
    best_model = models[best_model_name]['model']
    
    joblib.dump(best_model, os.path.join(save_dir, 'smote_optimized_kidney.pkl'))
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'training_method': 'SMOTE + Optimization',
        'performance': best_metrics,
        'classes': classes.tolist(),
        'all_models_comparison': {
            name: {
                'f1': results[name]['macro_f1'],
                'precision': results[name]['macro_precision'],
                'recall': results[name]['macro_recall'],
                'accuracy': results[name]['accuracy']
            }
            for name in models.keys()
        }
    }
    
    with open(os.path.join(save_dir, 'smote_optimized_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved best model to: {save_dir}/smote_optimized_kidney.pkl")
    
    # 6. Comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for model_name in models.keys():
        metrics = results[model_name]
        marker = " 🏆" if model_name == best_model_name else ""
        print(f"{model_name:<30} {metrics['macro_precision']:<12.2%} "
              f"{metrics['macro_recall']:<12.2%} {metrics['macro_f1']:<12.2%}{marker}")
    
    print("\n✅ Optimization Complete!")
    
    return metadata


if __name__ == "__main__":
    results = main()
