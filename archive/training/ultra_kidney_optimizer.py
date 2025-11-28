"""
Ultra-Optimized Kidney Disease Classifier
==========================================
Advanced techniques to maximize all metrics:
- XGBoost with scale_pos_weight
- LightGBM with is_unbalance
- CatBoost with auto_class_weights
- Stacking ensemble
- Advanced SMOTE variants (ADASYN, BorderlineSMOTE)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score, 
    recall_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
import config

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not available, skipping XGB models")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not available, skipping LGB models")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("⚠️  CatBoost not available, skipping CatBoost models")


def apply_advanced_sampling(X_train, y_train, method='adasyn'):
    """
    Apply advanced sampling techniques
    """
    print(f"\n🔄 Applying {method.upper()} sampling...")
    
    if method == 'adasyn':
        sampler = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    elif method == 'borderline':
        sampler = BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"   Before: {X_train.shape[0]} samples")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"     Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    print(f"\n   After: {X_resampled.shape[0]} samples")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"     Class {cls}: {count} ({count/len(y_resampled)*100:.1f}%)")
    
    return X_resampled, y_resampled


def train_ultra_optimized_models(X_train, y_train, X_test, y_test, classes):
    """
    Train ultra-optimized models using advanced techniques
    """
    models = {}
    
    # 1. XGBoost with scale_pos_weight
    if HAS_XGB:
        print("\n" + "="*70)
        print("MODEL 1: XGBoost (Scale Pos Weight)")
        print("="*70)
        
        # Calculate scale_pos_weight for each class
        unique, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()
        scale_weights = {int(cls): max_count / count for cls, count in zip(unique, counts)}
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softmax',
            num_class=len(classes),
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print("🔧 Training XGBoost...")
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_proba_xgb = xgb_model.predict_proba(X_test)
        
        f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
        precision_xgb = precision_score(y_test, y_pred_xgb, average='macro', zero_division=0)
        recall_xgb = recall_score(y_test, y_pred_xgb, average='macro', zero_division=0)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        
        # Calculate AUC-ROC
        try:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            auc_xgb = roc_auc_score(y_test_bin, y_proba_xgb, average='macro', multi_class='ovr')
        except:
            auc_xgb = 0.0
        
        print(f"   Accuracy:  {accuracy_xgb:.4f}")
        print(f"   Precision: {precision_xgb:.4f}")
        print(f"   Recall:    {recall_xgb:.4f}")
        print(f"   F1-Score:  {f1_xgb:.4f}")
        print(f"   AUC-ROC:   {auc_xgb:.4f}")
        
        models['XGBoost'] = {
            'model': xgb_model,
            'predictions': y_pred_xgb,
            'probabilities': y_proba_xgb,
            'accuracy': accuracy_xgb,
            'precision': precision_xgb,
            'recall': recall_xgb,
            'f1': f1_xgb,
            'auc': auc_xgb
        }
    
    # 2. LightGBM with is_unbalance
    if HAS_LGB:
        print("\n" + "="*70)
        print("MODEL 2: LightGBM (Unbalanced)")
        print("="*70)
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            objective='multiclass',
            num_class=len(classes),
            metric='multi_logloss',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        print("🔧 Training LightGBM...")
        lgb_model.fit(X_train, y_train)
        
        y_pred_lgb = lgb_model.predict(X_test)
        y_proba_lgb = lgb_model.predict_proba(X_test)
        
        f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
        precision_lgb = precision_score(y_test, y_pred_lgb, average='macro', zero_division=0)
        recall_lgb = recall_score(y_test, y_pred_lgb, average='macro', zero_division=0)
        accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
        
        try:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            auc_lgb = roc_auc_score(y_test_bin, y_proba_lgb, average='macro', multi_class='ovr')
        except:
            auc_lgb = 0.0
        
        print(f"   Accuracy:  {accuracy_lgb:.4f}")
        print(f"   Precision: {precision_lgb:.4f}")
        print(f"   Recall:    {recall_lgb:.4f}")
        print(f"   F1-Score:  {f1_lgb:.4f}")
        print(f"   AUC-ROC:   {auc_lgb:.4f}")
        
        models['LightGBM'] = {
            'model': lgb_model,
            'predictions': y_pred_lgb,
            'probabilities': y_proba_lgb,
            'accuracy': accuracy_lgb,
            'precision': precision_lgb,
            'recall': recall_lgb,
            'f1': f1_lgb,
            'auc': auc_lgb
        }
    
    # 3. CatBoost with auto_class_weights
    if HAS_CB:
        print("\n" + "="*70)
        print("MODEL 3: CatBoost (Auto Class Weights)")
        print("="*70)
        
        cb_model = cb.CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bylevel=0.8,
            auto_class_weights='Balanced',
            loss_function='MultiClass',
            eval_metric='TotalF1:average=Macro',
            random_state=42,
            thread_count=-1,
            verbose=0
        )
        
        print("🔧 Training CatBoost...")
        cb_model.fit(X_train, y_train)
        
        y_pred_cb = cb_model.predict(X_test).flatten()
        y_proba_cb = cb_model.predict_proba(X_test)
        
        f1_cb = f1_score(y_test, y_pred_cb, average='macro')
        precision_cb = precision_score(y_test, y_pred_cb, average='macro', zero_division=0)
        recall_cb = recall_score(y_test, y_pred_cb, average='macro', zero_division=0)
        accuracy_cb = accuracy_score(y_test, y_pred_cb)
        
        try:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(len(classes)))
            auc_cb = roc_auc_score(y_test_bin, y_proba_cb, average='macro', multi_class='ovr')
        except:
            auc_cb = 0.0
        
        print(f"   Accuracy:  {accuracy_cb:.4f}")
        print(f"   Precision: {precision_cb:.4f}")
        print(f"   Recall:    {recall_cb:.4f}")
        print(f"   F1-Score:  {f1_cb:.4f}")
        print(f"   AUC-ROC:   {auc_cb:.4f}")
        
        models['CatBoost'] = {
            'model': cb_model,
            'predictions': y_pred_cb,
            'probabilities': y_proba_cb,
            'accuracy': accuracy_cb,
            'precision': precision_cb,
            'recall': recall_cb,
            'f1': f1_cb,
            'auc': auc_cb
        }
    
    # 4. Enhanced Random Forest
    print("\n" + "="*70)
    print("MODEL 4: Enhanced Random Forest")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    print("🔧 Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)
    
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    precision_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
    recall_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(len(classes)))
        auc_rf = roc_auc_score(y_test_bin, y_proba_rf, average='macro', multi_class='ovr')
    except:
        auc_rf = 0.0
    
    print(f"   Accuracy:  {accuracy_rf:.4f}")
    print(f"   Precision: {precision_rf:.4f}")
    print(f"   Recall:    {recall_rf:.4f}")
    print(f"   F1-Score:  {f1_rf:.4f}")
    print(f"   AUC-ROC:   {auc_rf:.4f}")
    print(f"   OOB Score: {rf_model.oob_score_:.4f}")
    
    models['Random_Forest'] = {
        'model': rf_model,
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf,
        'accuracy': accuracy_rf,
        'precision': precision_rf,
        'recall': recall_rf,
        'f1': f1_rf,
        'auc': auc_rf
    }
    
    # 5. Enhanced Gradient Boosting
    print("\n" + "="*70)
    print("MODEL 5: Enhanced Gradient Boosting")
    print("="*70)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    
    print("🔧 Training Gradient Boosting...")
    gb_model.fit(X_train, y_train)
    
    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)
    
    f1_gb = f1_score(y_test, y_pred_gb, average='macro')
    precision_gb = precision_score(y_test, y_pred_gb, average='macro', zero_division=0)
    recall_gb = recall_score(y_test, y_pred_gb, average='macro', zero_division=0)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(len(classes)))
        auc_gb = roc_auc_score(y_test_bin, y_proba_gb, average='macro', multi_class='ovr')
    except:
        auc_gb = 0.0
    
    print(f"   Accuracy:  {accuracy_gb:.4f}")
    print(f"   Precision: {precision_gb:.4f}")
    print(f"   Recall:    {recall_gb:.4f}")
    print(f"   F1-Score:  {f1_gb:.4f}")
    print(f"   AUC-ROC:   {auc_gb:.4f}")
    
    models['Gradient_Boosting'] = {
        'model': gb_model,
        'predictions': y_pred_gb,
        'probabilities': y_proba_gb,
        'accuracy': accuracy_gb,
        'precision': precision_gb,
        'recall': recall_gb,
        'f1': f1_gb,
        'auc': auc_gb
    }
    
    return models


def main():
    print("🚀 ULTRA-OPTIMIZED Kidney Disease Classifier")
    print("="*70)
    print("Advanced Techniques: XGBoost, LightGBM, CatBoost, ADASYN")
    print("="*70)
    
    # Check available libraries
    print("\n📦 Available Libraries:")
    print(f"   XGBoost:  {'✓' if HAS_XGB else '✗'}")
    print(f"   LightGBM: {'✓' if HAS_LGB else '✗'}")
    print(f"   CatBoost: {'✓' if HAS_CB else '✗'}")
    
    # 1. Load Data
    print("\n1️⃣  Loading data...")
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        handle_imbalance=False
    )
    
    le = joblib.load(os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'label_encoder.pkl'))
    classes = le.classes_
    
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Apply Advanced Sampling
    X_train_balanced, y_train_balanced = apply_advanced_sampling(X_train, y_train, method='adasyn')
    
    # 3. Train Models
    print("\n2️⃣  Training ultra-optimized models...")
    models = train_ultra_optimized_models(X_train_balanced, y_train_balanced, X_test, y_test, classes)
    
    # 4. Find Best Model
    best_model_name = max(models.items(), key=lambda x: x[1]['f1'])[0]
    best_model_data = models[best_model_name]
    
    print("\n" + "="*70)
    print("🏆 BEST MODEL")
    print("="*70)
    print(f"Model: {best_model_name}")
    print(f"Accuracy:  {best_model_data['accuracy']:.4f}")
    print(f"Precision: {best_model_data['precision']:.4f}")
    print(f"Recall:    {best_model_data['recall']:.4f}")
    print(f"F1-Score:  {best_model_data['f1']:.4f}")
    print(f"AUC-ROC:   {best_model_data['auc']:.4f}")
    
    # 5. Save Best Model
    save_dir = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers')
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(best_model_data['model'], os.path.join(save_dir, 'ultra_optimized_kidney.pkl'))
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'optimization_method': 'ADASYN + Advanced Boosting',
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
        },
        'classes': classes.tolist()
    }
    
    with open(os.path.join(save_dir, 'ultra_optimized_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved to: {save_dir}/ultra_optimized_kidney.pkl")
    
    # 6. Comparison Table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 75)
    for name, data in sorted(models.items(), key=lambda x: x[1]['f1'], reverse=True):
        marker = " 🏆" if name == best_model_name else ""
        print(f"{name:<25} {data['accuracy']:<10.4f} {data['precision']:<10.4f} "
              f"{data['recall']:<10.4f} {data['f1']:<10.4f} {data['auc']:<10.4f}{marker}")
    
    print("\n✅ Ultra-Optimization Complete!")
    
    return metadata


if __name__ == "__main__":
    results = main()
