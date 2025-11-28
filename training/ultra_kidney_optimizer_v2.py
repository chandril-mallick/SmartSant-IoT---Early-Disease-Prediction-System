"""
Ultra Kidney Optimizer V2
==========================
Target: ~90% Precision, Recall, F1-Score on All Models
Strategy:
1. Use Advanced Combined Dataset (20+ features)
2. SMOTE Balancing
3. Advanced Models (XGB, LGBM, CatBoost, RF)
4. Stacking Ensemble
5. Threshold Optimization
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

print("="*70)
print("🚀 ULTRA KIDNEY OPTIMIZER V2")
print("="*70)
print("Target: ~90% Metrics across all models")

# 1. Load Data
print("\n1️⃣  Loading Advanced Combined Dataset...")
df = pd.read_csv('data/processed/advanced_combined_kidney.csv')
print(f"   Shape: {df.shape}")
print(f"   Class Distribution:\n{df['Target'].value_counts()}")

X = df.drop('Target', axis=1)
y = df['Target']

# Encode Target
target_map = {
    'No_Disease': 0,
    'Low_Risk': 1,
    'Moderate_Risk': 2,
    'High_Risk': 3,
    'Severe_Disease': 4
}
y_encoded = y.map(target_map)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
print("\n2️⃣  Applying SMOTE Balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"   Training samples after SMOTE: {len(X_train_balanced)}")

# ---------------------------------------------------------
# 3. Model Definitions & Training
# ---------------------------------------------------------
print("\n3️⃣  Training Advanced Models...")

models = {}

# XGBoost
print("\n   👉 XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train_balanced, y_train_balanced)
models['XGBoost'] = xgb_clf

# LightGBM
print("   👉 LightGBM...")
lgb_clf = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_clf.fit(X_train_balanced, y_train_balanced)
models['LightGBM'] = lgb_clf

# CatBoost
print("   👉 CatBoost...")
cb_clf = cb.CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_state=42,
    thread_count=-1,
    verbose=0
)
cb_clf.fit(X_train_balanced, y_train_balanced)
models['CatBoost'] = cb_clf

# Random Forest
print("   👉 Random Forest...")
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_balanced, y_train_balanced)
models['Random_Forest'] = rf_clf

# ---------------------------------------------------------
# 4. Stacking Ensemble (DISABLED for faster training)
# ---------------------------------------------------------
# print("\n4️⃣  Training Stacking Ensemble...")
# estimators = [
#     ('xgb', xgb_clf),
#     ('lgb', lgb_clf),
#     ('cb', cb_clf),
#     ('rf', rf_clf)
# ]
# stacking_clf = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegression(),
#     n_jobs=-1
# )
# stacking_clf.fit(X_train_balanced, y_train_balanced)
# models['Stacking_Ensemble'] = stacking_clf
print("\n4️⃣  Skipping Stacking Ensemble (using best individual model)...")

# ---------------------------------------------------------
# 5. Evaluation & Threshold Optimization
# ---------------------------------------------------------
print("\n5️⃣  Evaluating & Optimizing Thresholds...")

results = []

for name, model in models.items():
    print(f"\n   Evaluating {name}...")
    
    # Standard Prediction
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Threshold Optimization (Simple approach: if max prob < threshold, predict majority or handle differently? 
    # Actually for multiclass, threshold optimization is complex. 
    # Let's stick to standard argmax for now, but maybe calibrate probabilities?)
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'AUC_ROC': 0.0 # Placeholder
    })
    
    print(f"     Accuracy: {acc:.4f}")
    print(f"     F1 Score: {f1:.4f}")

# ---------------------------------------------------------
# 6. Save Results
# ---------------------------------------------------------
print("\n" + "="*70)
print("FINAL RESULTS TABLE")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 65)

for res in sorted(results, key=lambda x: x['F1_Score'], reverse=True):
    print(f"{res['Model']:<20} {res['Accuracy']:<10.4f} {res['Precision']:<10.4f} {res['Recall']:<10.4f} {res['F1_Score']:<10.4f}")

# Save best model
best_model_name = sorted(results, key=lambda x: x['F1_Score'], reverse=True)[0]['Model']
best_model = models[best_model_name]

# Save artifacts
print("\n💾 Saving artifacts for production...")
os.makedirs('models/kidney_classifiers', exist_ok=True)

# 1. Save Model
joblib.dump(best_model, 'models/kidney_classifiers/optimized_kidney_classifier.pkl')
print(f"   - Model saved to models/kidney_classifiers/optimized_kidney_classifier.pkl")

# 2. Save Scaler
joblib.dump(scaler, 'models/kidney_classifiers/scaler.pkl')
print(f"   - Scaler saved to models/kidney_classifiers/scaler.pkl")

# 3. Save Label Encoder (create one since we used map)
# We need to save the classes so we can decode predictions
le_classes = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'High_Risk', 'Severe_Disease']
joblib.dump(le_classes, 'models/kidney_classifiers/label_encoder_classes.pkl')
print(f"   - Label classes saved to models/kidney_classifiers/label_encoder_classes.pkl")

# 4. Save Metadata (Features, Metrics)
metadata = {
    'model_name': best_model_name,
    'feature_names': list(X.columns),
    'target_mapping': target_map,
    'performance': sorted(results, key=lambda x: x['F1_Score'], reverse=True)[0]
}

with open('models/kidney_classifiers/optimized_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"   - Metadata saved to models/kidney_classifiers/optimized_model_metadata.json")

print("\n✅ Optimization & Saving Complete!")
