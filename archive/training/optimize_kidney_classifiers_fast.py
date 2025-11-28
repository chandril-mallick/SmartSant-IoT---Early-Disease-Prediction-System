"""
Kidney Disease Classification - Optimized Version (Memory Efficient)
Reduced complexity for faster execution while still achieving ~90% performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, label_binarize

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


def optimize_models(X_train, X_test, y_train, y_test, num_classes, save_dir):
    """Optimize models with reduced complexity for faster execution."""
    print("\n" + "="*70)
    print("KIDNEY DISEASE CLASSIFIER OPTIMIZATION")
    print("="*70)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n" + "="*70)
    print("1. LOGISTIC REGRESSION")
    print("="*70)
    
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': ['balanced', None],
        'max_iter': [2000]
    }
    
    lr = LogisticRegression(random_state=42)
    lr_search = RandomizedSearchCV(
        lr, lr_params, n_iter=10, cv=5, scoring='f1_macro',
        n_jobs=-1, verbose=2, random_state=42
    )
    
    print("Training...")
    lr_search.fit(X_train, y_train)
    lr_model = lr_search.best_estimator_
    
    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)
    
    try:
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'params': lr_search.best_params_,
        'cv_score': lr_search.best_score_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'auc': auc
    }
    
    print(f"✅ Best CV F1: {lr_search.best_score_:.4f}")
    print(f"✅ Test Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print(f"✅ Test F1: {results['Logistic Regression']['f1']:.4f}")
    
    # 2. Random Forest (Reduced complexity)
    print("\n" + "="*70)
    print("2. RANDOM FOREST")
    print("="*70)
    
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=20, cv=5, scoring='f1_macro',
        n_jobs=-1, verbose=2, random_state=42
    )
    
    print("Training...")
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)
    
    try:
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_search.best_params_,
        'cv_score': rf_search.best_score_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'auc': auc
    }
    
    print(f"✅ Best CV F1: {rf_search.best_score_:.4f}")
    print(f"✅ Test Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"✅ Test F1: {results['Random Forest']['f1']:.4f}")
    
    # 3. Gradient Boosting
    print("\n" + "="*70)
    print("3. GRADIENT BOOSTING")
    print("="*70)
    
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_search = RandomizedSearchCV(
        gb, gb_params, n_iter=15, cv=5, scoring='f1_macro',
        n_jobs=-1, verbose=2, random_state=42
    )
    
    print("Training...")
    gb_search.fit(X_train, y_train)
    gb_model = gb_search.best_estimator_
    
    y_pred = gb_model.predict(X_test)
    y_proba = gb_model.predict_proba(X_test)
    
    try:
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'params': gb_search.best_params_,
        'cv_score': gb_search.best_score_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'auc': auc
    }
    
    print(f"✅ Best CV F1: {gb_search.best_score_:.4f}")
    print(f"✅ Test Accuracy: {results['Gradient Boosting']['accuracy']:.4f}")
    print(f"✅ Test F1: {results['Gradient Boosting']['f1']:.4f}")
    
    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "="*70)
        print("4. XGBOOST")
        print("="*70)
        
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        xgb_search = RandomizedSearchCV(
            xgb, xgb_params, n_iter=15, cv=5, scoring='f1_macro',
            n_jobs=-1, verbose=2, random_state=42
        )
        
        print("Training...")
        xgb_search.fit(X_train, y_train)
        xgb_model = xgb_search.best_estimator_
        
        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)
        
        try:
            y_test_bin = label_binarize(y_test, classes=range(num_classes))
            auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        except:
            auc = 0.0
        
        results['XGBoost'] = {
            'model': xgb_model,
            'params': xgb_search.best_params_,
            'cv_score': xgb_search.best_score_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'auc': auc
        }
        
        print(f"✅ Best CV F1: {xgb_search.best_score_:.4f}")
        print(f"✅ Test Accuracy: {results['XGBoost']['accuracy']:.4f}")
        print(f"✅ Test F1: {results['XGBoost']['f1']:.4f}")
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<25} {'CV F1':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<11} {'AUC':<10}")
    print("-" * 100)
    
    for name, result in results.items():
        print(f"{name:<25} {result['cv_score']:<10.4f} {result['accuracy']:<10.4f} "
              f"{result['precision']:<11.4f} {result['recall']:<10.4f} "
              f"{result['f1']:<11.4f} {result['auc']:<10.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_result = results[best_model_name]
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*70)
    print(f"   CV F1-Score: {best_result['cv_score']:.4f} ({best_result['cv_score']*100:.2f}%)")
    print(f"   Accuracy:    {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   Precision:   {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
    print(f"   Recall:      {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
    print(f"   F1-Score:    {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
    print(f"   AUC-ROC:     {best_result['auc']:.4f}")
    
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
            'cv_f1': float(best_result['cv_score']),
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
    
    # Save all results
    all_results_path = os.path.join(save_dir, 'all_models_comparison.json')
    all_results_data = {}
    for name, result in results.items():
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
                'cv_f1': float(result['cv_score']),
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1': float(result['f1']),
                'auc': float(result['auc'])
            }
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
    print("KIDNEY DISEASE CLASSIFIER OPTIMIZATION")
    print("Memory-Efficient Version - Target: ~90% Performance")
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
    
    # Run optimization
    save_dir = os.path.join(data_config.MODELS_DIR, 'kidney_classifiers')
    results, best_model = optimize_models(
        X_train, X_test, y_train_encoded, y_test_encoded, num_classes, save_dir
    )
    
    # Save label encoder
    le_path = os.path.join(save_dir, 'label_encoder.pkl')
    joblib.dump(le, le_path)
    print(f"   - Label encoder: label_encoder.pkl")
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
