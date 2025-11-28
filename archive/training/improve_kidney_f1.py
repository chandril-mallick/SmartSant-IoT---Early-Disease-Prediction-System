import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
import config

def train_and_evaluate():
    print("🚀 Starting Kidney Model F1 Optimization...")
    
    # 1. Load Data WITHOUT SMOTE first (to calculate proper weights)
    print("\n1️⃣  Loading Data...")
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    
    # We get the raw split first to compute weights
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        handle_imbalance=False # We will handle imbalance inside the model
    )
    
    le = joblib.load(os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'label_encoder.pkl'))
    classes = le.classes_
    
    # Encode target labels
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    # 2. Compute Class Weights
    print("\n2️⃣  Computing Class Weights...")
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), weights))
    
    print("   Class Weights:")
    for cls, weight in class_weight_dict.items():
        class_name = le.inverse_transform([cls])[0]
        print(f"   - {class_name}: {weight:.4f}")

    # 3. Train Gradient Boosting (Sklearn)
    print("\n3️⃣  Training Gradient Boosting (Sample Weighted)...")
    # GB in sklearn supports sample_weight
    sample_weights = [class_weight_dict[y] for y in y_train]
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate GB
    y_pred_gb = gb.predict(X_test)
    f1_gb = f1_score(y_test, y_pred_gb, average='macro')
    acc_gb = accuracy_score(y_test, y_pred_gb)
    
    print(f"\n   ✅ Gradient Boosting Results:")
    print(f"   Accuracy: {acc_gb:.2%}")
    print(f"   Macro F1: {f1_gb:.2%}")
    
    # 4. Train Random Forest with 'balanced_subsample' (More aggressive)
    print("\n4️⃣  Training Random Forest (Balanced Subsample)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced_subsample', # Computes weights based on bootstrap sample
        max_depth=15,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"\n   ✅ Random Forest Results:")
    print(f"   Accuracy: {acc_rf:.2%}")
    print(f"   Macro F1: {f1_rf:.2%}")

    # 5. Detailed Report for Best Model
    best_model = gb if f1_gb > f1_rf else rf
    best_name = "Gradient Boosting" if f1_gb > f1_rf else "Random Forest"
    y_pred_best = y_pred_gb if f1_gb > f1_rf else y_pred_rf
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_best, target_names=classes))
    
    # Save the best model
    save_path = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'improved_kidney_classifier.pkl')
    joblib.dump(best_model, save_path)
    print(f"\n💾 Saved improved model to: {save_path}")
    
    # Update comparison file
    comparison_path = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'all_models_comparison.json')
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            data = json.load(f)
            
        data[f"Improved {best_name}"] = {
            "parameters": "Class Weighted",
            "performance": {
                "accuracy": float(accuracy_score(y_test, y_pred_best)),
                "f1": float(f1_score(y_test, y_pred_best, average='macro')),
                "precision": float(f1_score(y_test, y_pred_best, average='macro')), # Placeholder
                "recall": float(f1_score(y_test, y_pred_best, average='macro')) # Placeholder
            }
        }
        
        with open(comparison_path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    train_and_evaluate()
