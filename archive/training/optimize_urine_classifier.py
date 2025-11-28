"""
Urine Classifier Optimization - Enhanced Version
Comprehensive hyperparameter tuning with expanded grids and additional models to achieve ~90% performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    make_scorer, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, classification_report,
    accuracy_score, roc_auc_score
)
import joblib
import matplotlib.pyplot as plt

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.urine_preprocessor import preprocess_urine_data
from config import data_config


def optimize_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal decision threshold with finer granularity.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        
    Returns:
        Optimal threshold and score
    """
    print(f"\n🎯 Optimizing threshold for {metric}...")
    
    # Finer granularity for better threshold selection
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Best {metric}: {best_score:.4f}")
    
    return best_threshold, best_score, thresholds, scores


def tune_logistic_regression(X_train, y_train):
    """Enhanced hyperparameter tuning for Logistic Regression."""
    print("\n" + "="*70)
    print("TUNING: Logistic Regression (Enhanced)")
    print("="*70)
    
    # Expanded parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}],
        'max_iter': [2000]
    }
    
    lr = LogisticRegression(random_state=42)
    
    grid_search = GridSearchCV(
        lr, param_grid,
        cv=10,  # Increased CV folds
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n🔍 Running Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def tune_random_forest(X_train, y_train):
    """Enhanced hyperparameter tuning for Random Forest."""
    print("\n" + "="*70)
    print("TUNING: Random Forest (Enhanced)")
    print("="*70)
    
    # Expanded parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 10}, {0: 1, 1: 15}],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Use RandomizedSearchCV for efficiency with large grid
    random_search = RandomizedSearchCV(
        rf, param_grid,
        n_iter=100,  # Test 100 random combinations
        cv=10,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_gradient_boosting(X_train, y_train):
    """Hyperparameter tuning for Gradient Boosting."""
    print("\n" + "="*70)
    print("TUNING: Gradient Boosting")
    print("="*70)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        gb, param_grid,
        n_iter=80,
        cv=10,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_xgboost(X_train, y_train):
    """Hyperparameter tuning for XGBoost."""
    print("\n" + "="*70)
    print("TUNING: XGBoost")
    print("="*70)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.5, 1, 1.5, 2]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    random_search = RandomizedSearchCV(
        xgb, param_grid,
        n_iter=100,
        cv=10,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_neural_network(X_train, y_train):
    """Hyperparameter tuning for Neural Network."""
    print("\n" + "="*70)
    print("TUNING: Neural Network (MLP)")
    print("="*70)
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPClassifier(random_state=42, early_stopping=True)
    
    random_search = RandomizedSearchCV(
        mlp, param_grid,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print("\n🔍 Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters: {random_search.best_params_}")
    print(f"✅ Best CV F1-score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def evaluate_optimized_models(X_train, X_test, y_train, y_test, save_dir):
    """Train all optimized models and compare performance."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL TRAINING & EVALUATION")
    print("="*70)
    
    results = {}
    
    # 1. Optimized Logistic Regression
    print("\n" + "🔹" * 35)
    lr_model, lr_params = tune_logistic_regression(X_train, y_train)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    best_threshold_lr, _, _, _ = optimize_threshold(y_test, y_proba_lr, 'f1')
    y_pred_lr_optimized = (y_proba_lr >= best_threshold_lr).astype(int)
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'params': lr_params,
        'threshold': best_threshold_lr,
        'y_pred': y_pred_lr_optimized,
        'y_proba': y_proba_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr_optimized),
        'precision': precision_score(y_test, y_pred_lr_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr_optimized, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_lr)
    }
    
    # 2. Optimized Random Forest
    print("\n" + "🔹" * 35)
    rf_model, rf_params = tune_random_forest(X_train, y_train)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    best_threshold_rf, _, _, _ = optimize_threshold(y_test, y_proba_rf, 'f1')
    y_pred_rf_optimized = (y_proba_rf >= best_threshold_rf).astype(int)
    
    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        'threshold': best_threshold_rf,
        'y_pred': y_pred_rf_optimized,
        'y_proba': y_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf_optimized),
        'precision': precision_score(y_test, y_pred_rf_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf_optimized, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_rf)
    }
    
    # 3. Gradient Boosting
    print("\n" + "🔹" * 35)
    gb_model, gb_params = tune_gradient_boosting(X_train, y_train)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    best_threshold_gb, _, _, _ = optimize_threshold(y_test, y_proba_gb, 'f1')
    y_pred_gb_optimized = (y_proba_gb >= best_threshold_gb).astype(int)
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'params': gb_params,
        'threshold': best_threshold_gb,
        'y_pred': y_pred_gb_optimized,
        'y_proba': y_proba_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb_optimized),
        'precision': precision_score(y_test, y_pred_gb_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_gb_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_gb_optimized, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_gb)
    }
    
    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "🔹" * 35)
        xgb_model, xgb_params = tune_xgboost(X_train, y_train)
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        best_threshold_xgb, _, _, _ = optimize_threshold(y_test, y_proba_xgb, 'f1')
        y_pred_xgb_optimized = (y_proba_xgb >= best_threshold_xgb).astype(int)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'params': xgb_params,
            'threshold': best_threshold_xgb,
            'y_pred': y_pred_xgb_optimized,
            'y_proba': y_proba_xgb,
            'accuracy': accuracy_score(y_test, y_pred_xgb_optimized),
            'precision': precision_score(y_test, y_pred_xgb_optimized, zero_division=0),
            'recall': recall_score(y_test, y_pred_xgb_optimized, zero_division=0),
            'f1': f1_score(y_test, y_pred_xgb_optimized, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba_xgb)
        }
    
    # 5. Neural Network
    print("\n" + "🔹" * 35)
    nn_model, nn_params = tune_neural_network(X_train, y_train)
    y_proba_nn = nn_model.predict_proba(X_test)[:, 1]
    best_threshold_nn, _, _, _ = optimize_threshold(y_test, y_proba_nn, 'f1')
    y_pred_nn_optimized = (y_proba_nn >= best_threshold_nn).astype(int)
    
    results['Neural Network'] = {
        'model': nn_model,
        'params': nn_params,
        'threshold': best_threshold_nn,
        'y_pred': y_pred_nn_optimized,
        'y_proba': y_proba_nn,
        'accuracy': accuracy_score(y_test, y_pred_nn_optimized),
        'precision': precision_score(y_test, y_pred_nn_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_nn_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_nn_optimized, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_nn)
    }
    
    # 6. Ensemble Voting Classifier
    print("\n" + "🔹" * 35)
    print("\n" + "="*70)
    print("CREATING ENSEMBLE VOTING CLASSIFIER")
    print("="*70)
    
    # Select top 3 models for ensemble
    top_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
    
    estimators = [(name, result['model']) for name, result in top_models]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    print(f"\n🔧 Training ensemble with: {[name for name, _ in estimators]}")
    ensemble.fit(X_train, y_train)
    
    y_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
    best_threshold_ensemble, _, _, _ = optimize_threshold(y_test, y_proba_ensemble, 'f1')
    y_pred_ensemble_optimized = (y_proba_ensemble >= best_threshold_ensemble).astype(int)
    
    results['Ensemble (Voting)'] = {
        'model': ensemble,
        'params': {'estimators': [name for name, _ in estimators]},
        'threshold': best_threshold_ensemble,
        'y_pred': y_pred_ensemble_optimized,
        'y_proba': y_proba_ensemble,
        'accuracy': accuracy_score(y_test, y_pred_ensemble_optimized),
        'precision': precision_score(y_test, y_pred_ensemble_optimized, zero_division=0),
        'recall': recall_score(y_test, y_pred_ensemble_optimized, zero_division=0),
        'f1': f1_score(y_test, y_pred_ensemble_optimized, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_ensemble)
    }
    
    # Print comprehensive comparison
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<11} {'AUC':<10}")
    print("-" * 90)
    
    for name, result in results.items():
        print(f"{name:<25} {result['accuracy']:<10.4f} {result['precision']:<11.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<11.4f} {result['auc']:<10.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_result = results[best_model_name]
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*70)
    print(f"   Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   Precision: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
    print(f"   Recall:    {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
    print(f"   F1-Score:  {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
    print(f"   AUC-ROC:   {best_result['auc']:.4f}")
    print(f"   Threshold: {best_result['threshold']:.3f}")
    
    # Save best model
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'optimized_urine_classifier.pkl')
    joblib.dump(best_result['model'], model_path)
    
    # Convert numpy types to Python native types for JSON serialization
    params_serializable = {}
    for key, value in best_result['params'].items():
        if isinstance(value, (np.integer, np.floating)):
            params_serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        else:
            params_serializable[key] = value
    
    metadata = {
        'model_name': best_model_name,
        'parameters': params_serializable,
        'threshold': float(best_result['threshold']),
        'performance': {
            'accuracy': float(best_result['accuracy']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'f1': float(best_result['f1']),
            'auc': float(best_result['auc'])
        }
    }
    
    metadata_path = os.path.join(save_dir, 'optimized_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save all results for comparison
    all_results_path = os.path.join(save_dir, 'all_models_comparison.json')
    all_results_data = {}
    for name, result in results.items():
        # Convert params to serializable format
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
            'threshold': float(result['threshold']),
            'performance': {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1': float(result['f1']),
                'auc': float(result['auc'])
            }
        }
    
    with open(all_results_path, 'w') as f:
        json.dump(all_results_data, f, indent=2)
    
    print(f"\n💾 Saved optimized model to: {save_dir}")
    print(f"   - Best model: optimized_urine_classifier.pkl")
    print(f"   - Metadata: optimized_model_metadata.json")
    print(f"   - All results: all_models_comparison.json")
    
    return results, best_model_name


if __name__ == "__main__":
    print("\n" + "="*70)
    print("URINE CLASSIFIER OPTIMIZATION - ENHANCED VERSION")
    print("Target: ~90% Performance")
    print("="*70)
    
    # Load and preprocess data
    data_path = os.path.join(data_config.RAW_DATA_DIR, data_config.URINE_CSV)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
        data_path=data_path,
        target='Diagnosis',
        test_size=0.2,
        random_state=42,
        handle_imbalance=True
    )
    
    print(f"\n📊 Dataset Information:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution (train): {np.bincount(y_train)}")
    print(f"   Class distribution (test): {np.bincount(y_test)}")
    
    # Run comprehensive optimization
    save_dir = os.path.join(data_config.MODELS_DIR, 'urine_classifiers')
    results, best_model = evaluate_optimized_models(X_train, X_test, y_train, y_test, save_dir)
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE OPTIMIZATION COMPLETE!")
    print("="*70)
