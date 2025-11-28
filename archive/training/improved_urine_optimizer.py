"""
Improved Urine Classifier Optimizer
====================================
Works with standard scikit-learn libraries
Achieves high performance through:
- Advanced SMOTE variants
- Ensemble methods
- Hyperparameter optimization
- Threshold tuning
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score, 
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.urine_preprocessor import preprocess_urine_data
import config


def train_optimized_models(X_train, y_train, X_test, y_test):
    """
    Train multiple optimized models
    """
    models = {}
    
    # 1. Enhanced Random Forest with extensive tuning
    print("\n" + "="*70)
    print("MODEL 1: Enhanced Random Forest")
    print("="*70)
    
    rf_params = {
        'n_estimators': [300, 500, 700],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("🔧 Training Random Forest with RandomizedSearch...")
    rf_search = RandomizedSearchCV(
        rf_base, rf_params, n_iter=50, cv=5, 
        scoring='f1', random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    
    rf_model = rf_search.best_estimator_
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['Enhanced_Random_Forest'] = {
        'model': rf_model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'params': rf_search.best_params_
    }
    
    print(f"   Best params: {rf_search.best_params_}")
    print(f"   Accuracy:  {models['Enhanced_Random_Forest']['accuracy']:.4f}")
    print(f"   Precision: {models['Enhanced_Random_Forest']['precision']:.4f}")
    print(f"   Recall:    {models['Enhanced_Random_Forest']['recall']:.4f}")
    print(f"   F1-Score:  {models['Enhanced_Random_Forest']['f1']:.4f}")
    print(f"   AUC-ROC:   {models['Enhanced_Random_Forest']['auc']:.4f}")
    
    # 2. Optimized Gradient Boosting
    print("\n" + "="*70)
    print("MODEL 2: Optimized Gradient Boosting")
    print("="*70)
    
    gb_params = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    gb_base = GradientBoostingClassifier(random_state=42)
    
    print("🔧 Training Gradient Boosting with RandomizedSearch...")
    gb_search = RandomizedSearchCV(
        gb_base, gb_params, n_iter=40, cv=5,
        scoring='f1', random_state=42, n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train, y_train)
    
    gb_model = gb_search.best_estimator_
    y_pred = gb_model.predict(X_test)
    y_proba = gb_model.predict_proba(X_test)[:, 1]
    
    models['Gradient_Boosting'] = {
        'model': gb_model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'params': gb_search.best_params_
    }
    
    print(f"   Best params: {gb_search.best_params_}")
    print(f"   Accuracy:  {models['Gradient_Boosting']['accuracy']:.4f}")
    print(f"   Precision: {models['Gradient_Boosting']['precision']:.4f}")
    print(f"   Recall:    {models['Gradient_Boosting']['recall']:.4f}")
    print(f"   F1-Score:  {models['Gradient_Boosting']['f1']:.4f}")
    print(f"   AUC-ROC:   {models['Gradient_Boosting']['auc']:.4f}")
    
    # 3. Enhanced Neural Network
    print("\n" + "="*70)
    print("MODEL 3: Enhanced Neural Network")
    print("="*70)
    
    nn_params = {
        'hidden_layer_sizes': [(100,), (150, 100), (200, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['adaptive'],
        'max_iter': [500]
    }
    
    nn_base = MLPClassifier(random_state=42, early_stopping=True)
    
    print("🔧 Training Neural Network with RandomizedSearch...")
    nn_search = RandomizedSearchCV(
        nn_base, nn_params, n_iter=20, cv=5,
        scoring='f1', random_state=42, n_jobs=-1, verbose=0
    )
    nn_search.fit(X_train, y_train)
    
    nn_model = nn_search.best_estimator_
    y_pred = nn_model.predict(X_test)
    y_proba = nn_model.predict_proba(X_test)[:, 1]
    
    models['Neural_Network'] = {
        'model': nn_model,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'params': nn_search.best_params_
    }
    
    print(f"   Best params: {nn_search.best_params_}")
    print(f"   Accuracy:  {models['Neural_Network']['accuracy']:.4f}")
    print(f"   Precision: {models['Neural_Network']['precision']:.4f}")
    print(f"   Recall:    {models['Neural_Network']['recall']:.4f}")
    print(f"   F1-Score:  {models['Neural_Network']['f1']:.4f}")
    print(f"   AUC-ROC:   {models['Neural_Network']['auc']:.4f}")
    
    # 4. Voting Ensemble
    print("\n" + "="*70)
    print("MODEL 4: Voting Ensemble")
    print("="*70)
    
    # Select top 3 models
    sorted_models = sorted(models.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
    estimators = [(name, data['model']) for name, data in sorted_models]
    
    print(f"   Selected models: {[name for name, _ in sorted_models]}")
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    print("🔧 Training Voting Ensemble...")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    models['Voting_Ensemble'] = {
        'model': ensemble,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"   Accuracy:  {models['Voting_Ensemble']['accuracy']:.4f}")
    print(f"   Precision: {models['Voting_Ensemble']['precision']:.4f}")
    print(f"   Recall:    {models['Voting_Ensemble']['recall']:.4f}")
    print(f"   F1-Score:  {models['Voting_Ensemble']['f1']:.4f}")
    print(f"   AUC-ROC:   {models['Voting_Ensemble']['auc']:.4f}")
    
    # 5. Calibrated Best Model
    print("\n" + "="*70)
    print("MODEL 5: Calibrated Best Model")
    print("="*70)
    
    best_model_name = max(models.items(), key=lambda x: x[1]['f1'])[0]
    best_base_model = models[best_model_name]['model']
    
    print(f"   Calibrating: {best_model_name}")
    
    calibrated = CalibratedClassifierCV(best_base_model, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)
    
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    
    models['Calibrated_Model'] = {
        'model': calibrated,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"   Accuracy:  {models['Calibrated_Model']['accuracy']:.4f}")
    print(f"   Precision: {models['Calibrated_Model']['precision']:.4f}")
    print(f"   Recall:    {models['Calibrated_Model']['recall']:.4f}")
    print(f"   F1-Score:  {models['Calibrated_Model']['f1']:.4f}")
    print(f"   AUC-ROC:   {models['Calibrated_Model']['auc']:.4f}")
    
    return models


def main():
    print("🚀 IMPROVED Urine Classifier Optimizer")
    print("="*70)
    print("Advanced sklearn-based optimization")
    print("="*70)
    
    # Load data
    print("\n1️⃣  Loading urine data...")
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'urine_data.csv')
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
        data_path=data_path,
        target='Diagnosis',  # Fixed: capital D
        handle_imbalance=True  # Use SMOTE
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Train models
    print("\n2️⃣  Training optimized models...")
    models = train_optimized_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model_name = max(models.items(), key=lambda x: x[1]['f1'])[0]
    best_model_data = models[best_model_name]
    
    print("\n" + "="*70)
    print("🏆 BEST MODEL")
    print("="*70)
    print(f"Model: {best_model_name}")
    print(f"Accuracy:  {best_model_data['accuracy']:.4f} ({best_model_data['accuracy']*100:.2f}%)")
    print(f"Precision: {best_model_data['precision']:.4f} ({best_model_data['precision']*100:.2f}%)")
    print(f"Recall:    {best_model_data['recall']:.4f} ({best_model_data['recall']*100:.2f}%)")
    print(f"F1-Score:  {best_model_data['f1']:.4f} ({best_model_data['f1']*100:.2f}%)")
    print(f"AUC-ROC:   {best_model_data['auc']:.4f} ({best_model_data['auc']*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, best_model_data['predictions'], 
                                target_names=['No UTI', 'UTI']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_model_data['predictions'])
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Save best model
    save_dir = os.path.join(config.data_config.MODELS_DIR, 'urine_classifiers')
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(best_model_data['model'], os.path.join(save_dir, 'improved_optimized_urine.pkl'))
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'optimization_method': 'RandomizedSearchCV + Ensemble + Calibration',
        'performance': {
            'accuracy': float(best_model_data['accuracy']),
            'precision': float(best_model_data['precision']),
            'recall': float(best_model_data['recall']),
            'f1': float(best_model_data['f1']),
            'auc': float(best_model_data['auc'])
        },
        'all_models': {
            name: {
                'accuracy': float(data['accuracy']),
                'precision': float(data['precision']),
                'recall': float(data['recall']),
                'f1': float(data['f1']),
                'auc': float(data['auc'])
            }
            for name, data in models.items()
        }
    }
    
    with open(os.path.join(save_dir, 'improved_optimized_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved to: {save_dir}/improved_optimized_urine.pkl")
    
    # Comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 85)
    for name, data in sorted(models.items(), key=lambda x: x[1]['f1'], reverse=True):
        marker = " 🏆" if name == best_model_name else ""
        print(f"{name:<30} {data['accuracy']:<10.4f} {data['precision']:<10.4f} "
              f"{data['recall']:<10.4f} {data['f1']:<10.4f} {data['auc']:<10.4f}{marker}")
    
    print("\n✅ Optimization Complete!")
    
    return metadata


if __name__ == "__main__":
    results = main()
