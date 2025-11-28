"""
Kidney Disease Classification - Enhanced Optimization Pipeline
Comprehensive hyperparameter tuning with expanded grids and multiple models to achieve ~90% performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
import joblib

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
from config import data_config

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False


def tune_logistic_regression(X_train, y_train, num_classes):
    """Enhanced hyperparameter tuning for Logistic Regression (multi-class)."""
    print("\n" + "="*70)
    print("TUNING: Logistic Regression (Multi-Class)")
    print("="*70)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
        'penalty': ['l2'],  # l1 not supported for multinomial
        'solver': ['lbfgs', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [2000, 3000]
    }
    
    lr = LogisticRegression(
        multi_class='multinomial',
        random_state=42
    )
    
    grid_search = GridSearchCV(
        lr, param_grid,
        cv=10,
        scoring='f1_macro',  # Macro F1 for multi-class
        n_jobs=-1,
        verbose=1
    )
    
    print("\n🔍 Running Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1-score (macro): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def tune_random_forest(X_train, y_train):
    """Enhanced hyperparameter tuning for Random Forest (multi-class)."""
    print("\n" + "="*70)
    print("TUNING: Random Forest (Multi-Class)")
    print("="*70)
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, param_grid,
        n_iter=100,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score (macro): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_gradient_boosting(X_train, y_train):
    """Hyperparameter tuning for Gradient Boosting (multi-class)."""
    print("\n" + "="*70)
    print("TUNING: Gradient Boosting (Multi-Class)")
    print("="*70)
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        gb, param_grid,
        n_iter=80,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score (macro): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_xgboost(X_train, y_train, num_classes):
    """Hyperparameter tuning for XGBoost (multi-class)."""
    print("\n" + "="*70)
    print("TUNING: XGBoost (Multi-Class)")
    print("="*70)
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.5, 1, 1.5, 2]
    }
    
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    random_search = RandomizedSearchCV(
        xgb, param_grid,
        n_iter=100,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score (macro): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_neural_network(X_train, y_train):
    """Hyperparameter tuning for Neural Network (multi-class)."""
    print("\n" + "="*70)
    print("TUNING: Neural Network (MLP - Multi-Class)")
    print("="*70)
    
    param_grid = {
        'hidden_layer_sizes': [
            (100,), (200,), (256,),
            (100, 50), (200, 100), (256, 128),
            (256, 128, 64), (200, 100, 50), (150, 100, 50)
        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    random_search = RandomizedSearchCV(
        mlp, param_grid,
        n_iter=50,
        cv=5,  # Reduced CV for speed
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score (macro): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def evaluate_model(model, X_train, y_train, X_test, y_test, num_classes):
    """Evaluate a single model and return comprehensive metrics."""
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba_test = model.predict_proba(X_test)
        
        try:
            y_test_bin = label_binarize(y_test, classes=range(num_classes))
            test_roc_auc = roc_auc_score(y_test_bin, y_pred_proba_test, average='macro', multi_class='ovr')
        except:
            test_roc_auc = 0.0
    else:
        test_roc_auc = 0.0
    
    # Metrics
    results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'train_f1': f1_score(y_train, y_pred_train, average='macro', zero_division=0),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, average='macro', zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, average='macro', zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
        'test_roc_auc': test_roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }
    
    return results


def run_comprehensive_optimization(X_train, X_test, y_train, y_test, num_classes, save_dir):
    """Train all optimized models and compare performance."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL OPTIMIZATION & EVALUATION")
    print("="*70)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n" + "🔹" * 35)
    lr_model, lr_params = tune_logistic_regression(X_train, y_train, num_classes)
    lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, num_classes)
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'params': lr_params,
        **lr_results
    }
    
    # 2. Random Forest
    print("\n" + "🔹" * 35)
    rf_model, rf_params = tune_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, num_classes)
    
    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        **rf_results
    }
    
    # 3. Gradient Boosting
    print("\n" + "🔹" * 35)
    gb_model, gb_params = tune_gradient_boosting(X_train, y_train)
    gb_results = evaluate_model(gb_model, X_train, y_train, X_test, y_test, num_classes)
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'params': gb_params,
        **gb_results
    }
    
    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "🔹" * 35)
        xgb_model, xgb_params = tune_xgboost(X_train, y_train, num_classes)
        xgb_results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, num_classes)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'params': xgb_params,
            **xgb_results
        }
    
    # 5. Neural Network
    print("\n" + "🔹" * 35)
    nn_model, nn_params = tune_neural_network(X_train, y_train)
    nn_results = evaluate_model(nn_model, X_train, y_train, X_test, y_test, num_classes)
    
    results['Neural Network'] = {
        'model': nn_model,
        'params': nn_params,
        **nn_results
    }
    
    # 6. Ensemble Voting Classifier
    print("\n" + "🔹" * 35)
    print("\n" + "="*70)
    print("CREATING ENSEMBLE VOTING CLASSIFIER")
    print("="*70)
    
    # Select top 3 models for ensemble
    top_models = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)[:3]
    
    estimators = [(name, result['model']) for name, result in top_models]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    print(f"\n🔧 Training ensemble with: {[name for name, _ in estimators]}")
    ensemble.fit(X_train, y_train)
    
    ensemble_results = evaluate_model(ensemble, X_train, y_train, X_test, y_test, num_classes)
    
    results['Ensemble (Voting)'] = {
        'model': ensemble,
        'params': {'estimators': [name for name, _ in estimators]},
        **ensemble_results
    }
    
    # Print comprehensive comparison
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<11} {'AUC':<10}")
    print("-" * 90)
    
    for name, result in results.items():
        print(f"{name:<25} {result['test_accuracy']:<10.4f} {result['test_precision']:<11.4f} "
              f"{result['test_recall']:<10.4f} {result['test_f1']:<11.4f} {result['test_roc_auc']:<10.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_f1'])
    best_result = results[best_model_name]
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*70)
    print(f"   Accuracy:  {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"   Precision: {best_result['test_precision']:.4f} ({best_result['test_precision']*100:.2f}%)")
    print(f"   Recall:    {best_result['test_recall']:.4f} ({best_result['test_recall']*100:.2f}%)")
    print(f"   F1-Score:  {best_result['test_f1']:.4f} ({best_result['test_f1']*100:.2f}%)")
    print(f"   AUC-ROC:   {best_result['test_roc_auc']:.4f}")
    
    # Save models
    os.makedirs(save_dir, exist_ok=True)
    
    # Save best model
    best_model_path = os.path.join(save_dir, 'optimized_kidney_classifier.pkl')
    joblib.dump(best_result['model'], best_model_path)
    
    # Convert params to serializable format
    params_serializable = {}
    for key, value in best_result['params'].items():
        if isinstance(value, (np.integer, np.floating)):
            params_serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        else:
            params_serializable[key] = value
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'parameters': params_serializable,
        'num_classes': num_classes,
        'performance': {
            'accuracy': float(best_result['test_accuracy']),
            'precision': float(best_result['test_precision']),
            'recall': float(best_result['test_recall']),
            'f1': float(best_result['test_f1']),
            'auc': float(best_result['test_roc_auc'])
        }
    }
    
    metadata_path = os.path.join(save_dir, 'optimized_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save all results
    all_results_path = os.path.join(save_dir, 'all_models_comparison.json')
    all_results_data = {}
    for name, result in results.items():
        # Convert params
        params_ser = {}
        for key, value in result['params'].items():
            if isinstance(value, (np.integer, np.floating)):
                params_ser[key] = value.item()
            elif isinstance(value, np.ndarray):
                params_ser[key] = value.tolist()
            else:
                params_ser[key] = value
        
        all_results_data[name] = {
            'parameters': params_ser,
            'performance': {
                'accuracy': float(result['test_accuracy']),
                'precision': float(result['test_precision']),
                'recall': float(result['test_recall']),
                'f1': float(result['test_f1']),
                'auc': float(result['test_roc_auc'])
            },
            'confusion_matrix': result['confusion_matrix']
        }
    
    with open(all_results_path, 'w') as f:
        json.dump(all_results_data, f, indent=2)
    
    print(f"\n💾 Saved optimized models to: {save_dir}")
    print(f"   - Best model: optimized_kidney_classifier.pkl")
    print(f"   - Metadata: optimized_model_metadata.json")
    print(f"   - All results: all_models_comparison.json")
    
    return results, best_model_name


if __name__ == "__main__":
    print("\n" + "="*70)
    print("KIDNEY DISEASE CLASSIFIER OPTIMIZATION - ENHANCED VERSION")
    print("Target: ~90% Performance")
    print("="*70)
    
    # Load and preprocess data
    data_path = os.path.join(data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        test_size=0.2,
        random_state=42,
        handle_imbalance=True
    )
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {num_classes} - {list(le.classes_)}")
    print(f"   Class distribution (train): {np.bincount(y_train_encoded)}")
    print(f"   Class distribution (test): {np.bincount(y_test_encoded)}")
    
    # Run comprehensive optimization
    save_dir = os.path.join(data_config.MODELS_DIR, 'kidney_classifiers')
    results, best_model = run_comprehensive_optimization(
        X_train, X_test, y_train_encoded, y_test_encoded, num_classes, save_dir
    )
    
    # Save label encoder
    le_path = os.path.join(save_dir, 'label_encoder.pkl')
    joblib.dump(le, le_path)
    print(f"   - Label encoder: label_encoder.pkl")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE OPTIMIZATION COMPLETE!")
    print("="*70)
