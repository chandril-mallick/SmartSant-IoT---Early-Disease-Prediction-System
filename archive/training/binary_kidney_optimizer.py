"""
Binary Kidney Optimizer (Healthy vs Disease)
=============================================
Target: >90% Metrics on Binary Classification
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

print("="*70)
print("🚀 BINARY KIDNEY OPTIMIZER (HEALTHY VS DISEASE)")
print("="*70)

# 1. Load Data
df = pd.read_csv('data/processed/advanced_combined_kidney.csv')

# 2. Binarize Target
# No_Disease -> 0
# Low_Risk, Moderate_Risk, High_Risk, Severe_Disease -> 1
df['Binary_Target'] = df['Target'].apply(lambda x: 0 if x == 'No_Disease' else 1)

print(f"Class Distribution:\n{df['Binary_Target'].value_counts()}")

X = df.drop(['Target', 'Binary_Target'], axis=1)
y = df['Binary_Target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train Models
models = {}

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_clf.fit(X_train_balanced, y_train_balanced)
models['XGBoost'] = xgb_clf

# LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
lgb_clf.fit(X_train_balanced, y_train_balanced)
models['LightGBM'] = lgb_clf

# CatBoost
cb_clf = cb.CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05, verbose=0, random_state=42)
cb_clf.fit(X_train_balanced, y_train_balanced)
models['CatBoost'] = cb_clf

# Evaluate
print("\n" + "="*70)
print("RESULTS (BINARY)")
print("="*70)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
print("-" * 75)

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"{name:<15} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")

# Save Best
best_model_name = "LightGBM" # Placeholder
joblib.dump(models['LightGBM'], 'models/kidney_classifiers/binary_best_model.pkl')
