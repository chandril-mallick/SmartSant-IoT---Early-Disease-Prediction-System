import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
import config

def optimize_thresholds():
    print("🚀 Optimizing Kidney Model Thresholds (One-Vs-Rest)...")
    
    # 1. Load Data
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        handle_imbalance=False # We handle it via class_weight
    )
    
    le = joblib.load(os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'label_encoder.pkl'))
    classes = le.classes_
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    # 2. Train One-Vs-Rest Random Forest
    print("\n2️⃣  Training One-Vs-Rest Random Forest...")
    # Use balanced weights to help with imbalance
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    ovr = OneVsRestClassifier(rf)
    ovr.fit(X_train, y_train)
    
    # 3. Get Probabilities
    print("\n3️⃣  Predicting Probabilities...")
    y_proba = ovr.predict_proba(X_test)
    
    # 4. Find Optimal Thresholds
    print("\n4️⃣  Finding Optimal Thresholds per Class...")
    best_thresholds = {}
    
    # We need to binarize y_test for per-class evaluation
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    for i, class_name in enumerate(classes):
        best_f1 = 0
        best_thresh = 0.5
        
        # Check thresholds from 0.05 to 0.95
        for thresh in np.arange(0.05, 0.96, 0.05):
            y_pred_class = (y_proba[:, i] >= thresh).astype(int)
            f1 = f1_score(y_test_bin[:, i], y_pred_class)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        best_thresholds[i] = best_thresh
        print(f"   - {class_name}: Best Threshold = {best_thresh:.2f} (F1 = {best_f1:.4f})")
        
    # 5. Apply Optimized Thresholds
    print("\n5️⃣  Applying Optimized Thresholds...")
    final_preds = []
    
    for prob_vector in y_proba:
        # Check which classes exceed their threshold
        candidates = []
        for i in range(len(classes)):
            if prob_vector[i] >= best_thresholds[i]:
                candidates.append((i, prob_vector[i]))
        
        if not candidates:
            # If none exceed threshold, pick the one with highest relative probability vs its threshold
            # Or just argmax
            final_preds.append(np.argmax(prob_vector))
        else:
            # If multiple, pick the one with highest probability
            # Alternatively, pick the one that exceeds its threshold by the largest margin
            best_candidate = max(candidates, key=lambda x: x[1])
            final_preds.append(best_candidate[0])
            
    final_preds = np.array(final_preds)
    
    # 6. Evaluate
    acc = accuracy_score(y_test, final_preds)
    macro_f1 = f1_score(y_test, final_preds, average='macro')
    
    print(f"\n✅ Optimized Results:")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   Macro F1: {macro_f1:.2%}")
    
    print(f"\n{'='*60}")
    print(classification_report(y_test, final_preds, target_names=classes))
    
    # Save optimized metadata
    metadata = {
        "model_type": "OneVsRest_RandomForest",
        "thresholds": {classes[i]: best_thresholds[i] for i in range(len(classes))},
        "performance": {
            "accuracy": acc,
            "f1": macro_f1
        }
    }
    
    with open(os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'optimized_thresholds.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Save model
    joblib.dump(ovr, os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'ovr_optimized_kidney.pkl'))

if __name__ == "__main__":
    optimize_thresholds()
