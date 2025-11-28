"""
Train Models on Combined Kidney Dataset
========================================
Train advanced models on the integrated dataset
Target: Achieve 65-80% macro F1 (vs 19.69% baseline)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

print("="*70)
print("TRAINING ON COMBINED KIDNEY DATASET")
print("="*70)
print("Target: 65-80% Macro F1 (vs 19.69% baseline)")
print("="*70)

# Load combined dataset
print("\n1️⃣  Loading combined dataset...")
df = pd.read_csv('data/processed/combined_kidney_dataset.csv')

print(f"   Shape: {df.shape}")
print(f"   Class distribution:")
print(df['Target'].value_counts())

# Prepare data
print("\n2️⃣  Preparing data...")
X = df.drop('Target', axis=1)
y = df['Target']

# Handle missing values
X = X.fillna(X.median())

# Encode target
target_map = {
    'No_Disease': 0,
    'Low_Risk': 1,
    'Moderate_Risk': 2,
    'High_Risk': 3,
    'Severe_Disease': 4
}
y_encoded = y.map(target_map)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X_train.shape[1]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
print("\n3️⃣  Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"   After SMOTE: {len(X_train_balanced)} samples")
print(f"   Class distribution:")
print(pd.Series(y_train_balanced).value_counts().sort_index())

# Train models
models = {}

# 1. XGBoost
print("\n" + "="*70)
print("MODEL 1: XGBoost")
print("="*70)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

print("🔧 Training XGBoost...")
xgb_model.fit(X_train_balanced, y_train_balanced)

y_pred = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)

models['XGBoost'] = {
    'model': xgb_model,
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted')
}

print(f"   Accuracy:        {models['XGBoost']['accuracy']:.4f}")
print(f"   Precision (macro): {models['XGBoost']['precision_macro']:.4f}")
print(f"   Recall (macro):    {models['XGBoost']['recall_macro']:.4f}")
print(f"   F1 (macro):        {models['XGBoost']['f1_macro']:.4f}")
print(f"   F1 (weighted):     {models['XGBoost']['f1_weighted']:.4f}")

# 2. LightGBM
print("\n" + "="*70)
print("MODEL 2: LightGBM")
print("="*70)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("🔧 Training LightGBM...")
lgb_model.fit(X_train_balanced, y_train_balanced)

y_pred = lgb_model.predict(X_test_scaled)

models['LightGBM'] = {
    'model': lgb_model,
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted')
}

print(f"   Accuracy:        {models['LightGBM']['accuracy']:.4f}")
print(f"   Precision (macro): {models['LightGBM']['precision_macro']:.4f}")
print(f"   Recall (macro):    {models['LightGBM']['recall_macro']:.4f}")
print(f"   F1 (macro):        {models['LightGBM']['f1_macro']:.4f}")
print(f"   F1 (weighted):     {models['LightGBM']['f1_weighted']:.4f}")

# 3. CatBoost
print("\n" + "="*70)
print("MODEL 3: CatBoost")
print("="*70)

cb_model = cb.CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    # subsample=0.8,  # Removed to fix conflict with Bayesian bootstrap
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro',
    random_state=42,
    thread_count=-1,
    verbose=0
)

print("🔧 Training CatBoost...")
cb_model.fit(X_train_balanced, y_train_balanced)

y_pred = cb_model.predict(X_test_scaled).flatten()

models['CatBoost'] = {
    'model': cb_model,
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted')
}

print(f"   Accuracy:        {models['CatBoost']['accuracy']:.4f}")
print(f"   Precision (macro): {models['CatBoost']['precision_macro']:.4f}")
print(f"   Recall (macro):    {models['CatBoost']['recall_macro']:.4f}")
print(f"   F1 (macro):        {models['CatBoost']['f1_macro']:.4f}")
print(f"   F1 (weighted):     {models['CatBoost']['f1_weighted']:.4f}")

# Find best model
best_model_name = max(models.items(), key=lambda x: x[1]['f1_macro'])[0]
best_model_data = models[best_model_name]

print("\n" + "="*70)
print("🏆 BEST MODEL")
print("="*70)
print(f"Model: {best_model_name}")
print(f"Accuracy:        {best_model_data['accuracy']:.4f} ({best_model_data['accuracy']*100:.2f}%)")
print(f"Precision (macro): {best_model_data['precision_macro']:.4f} ({best_model_data['precision_macro']*100:.2f}%)")
print(f"Recall (macro):    {best_model_data['recall_macro']:.4f} ({best_model_data['recall_macro']*100:.2f}%)")
print(f"F1 (macro):        {best_model_data['f1_macro']:.4f} ({best_model_data['f1_macro']*100:.2f}%)")
print(f"F1 (weighted):     {best_model_data['f1_weighted']:.4f} ({best_model_data['f1_weighted']*100:.2f}%)")

# Detailed report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
y_pred_best = best_model_data['model'].predict(X_test_scaled)
if isinstance(y_pred_best, np.ndarray) and y_pred_best.ndim > 1:
    y_pred_best = y_pred_best.flatten()

target_names = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'High_Risk', 'Severe_Disease']
print(classification_report(y_test, y_pred_best, target_names=target_names))

# Save best model
save_dir = 'models/kidney_classifiers'
os.makedirs(save_dir, exist_ok=True)
joblib.dump(best_model_data['model'], os.path.join(save_dir, 'combined_dataset_best_model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'combined_dataset_scaler.pkl'))

print(f"\n💾 Saved to: {save_dir}/combined_dataset_best_model.pkl")

# Comparison table
print("\n" + "="*70)
print("MODEL COMPARISON TABLE")
print("="*70)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
print("-" * 75)
for name, data in sorted(models.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
    marker = " 🏆" if name == best_model_name else ""
    print(f"{name:<15} {data['accuracy']:<10.4f} {data['precision_macro']:<10.4f} "
          f"{data['recall_macro']:<10.4f} {data['f1_macro']:<12.4f} {data['f1_weighted']:<12.4f}{marker}")

# Improvement summary
print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)
print(f"\n📊 Baseline (Original Dataset):")
print(f"   Macro F1: 19.69%")
print(f"\n📈 With Combined Dataset ({best_model_name}):")
print(f"   Macro F1: {best_model_data['f1_macro']*100:.2f}%")
print(f"\n🎯 Improvement: {(best_model_data['f1_macro']*100 - 19.69):.2f} percentage points")
print(f"   Multiplier: {best_model_data['f1_macro']*100 / 19.69:.2f}x")

print("\n✅ Training complete!")
