import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.kidney_preprocessor import preprocess_kidney_data
import config

def calculate_metrics(model, X_test, y_test, le_classes):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Macro average for multi-class
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    # AUC (One-vs-Rest)
    try:
        y_test_bin = label_binarize(y_test, classes=range(len(le_classes)))
        metrics['auc'] = float(np.mean([
            roc_auc_score(y_test_bin[:, i], y_proba[:, i]) 
            for i in range(len(le_classes))
        ]))
    except:
        metrics['auc'] = 0.0
        
    return metrics

def main():
    print("🚀 Training Missing Kidney Models (Baseline LR, NN, Voting)...")
    
    # Load Data
    data_path = os.path.join(config.data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        handle_imbalance=True
    )
    
    # Load Label Encoder
    le_path = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'label_encoder.pkl')
    le = joblib.load(le_path)
    
    # Load existing results to get best params for RF and GB
    results_path = os.path.join(config.data_config.MODELS_DIR, 'kidney_classifiers', 'all_models_comparison.json')
    with open(results_path, 'r') as f:
        existing_results = json.load(f)
        
    rf_params = existing_results['Random Forest']['parameters']
    gb_params = existing_results['Gradient Boosting']['parameters']
    
    # 1. Baseline Logistic Regression (Default params)
    print("\nTraining Baseline Logistic Regression...")
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train, y_train)
    baseline_metrics = calculate_metrics(baseline_lr, X_test, y_test, le.classes_)
    
    # 2. Neural Network
    print("\nTraining Neural Network...")
    # Using a standard robust architecture
    nn = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    nn.fit(X_train, y_train)
    nn_metrics = calculate_metrics(nn, X_test, y_test, le.classes_)
    
    # Calculate CV for NN
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nn_cv_scores = cross_val_score(nn, X_train, y_train, cv=cv, scoring='f1_macro')
    nn_metrics['cv_f1'] = nn_cv_scores.mean()
    
    # 3. Voting Classifier
    print("\nTraining Voting Classifier...")
    # Re-instantiate best models
    rf = RandomForestClassifier(**rf_params, random_state=42)
    gb = GradientBoostingClassifier(**gb_params, random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('nn', nn)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    voting_metrics = calculate_metrics(voting_clf, X_test, y_test, le.classes_)
    
    # Calculate CV for Voting (might be slow, so maybe skip or do fewer folds)
    # We'll skip CV for voting to save time, or do 3 folds
    voting_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=3, scoring='f1_macro')
    voting_metrics['cv_f1'] = voting_cv_scores.mean()
    
    # Update Results
    new_results = {
        "Baseline LR": {
            "parameters": "default",
            "performance": baseline_metrics
        },
        "Neural Network": {
            "parameters": nn.get_params(),
            "performance": nn_metrics
        },
        "Ensemble (Voting)": {
            "parameters": "RF+GB+NN (Soft Voting)",
            "performance": voting_metrics
        }
    }
    
    existing_results.update(new_results)
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
        
    print("\n✅ Updated all_models_comparison.json with missing models.")

if __name__ == "__main__":
    main()
