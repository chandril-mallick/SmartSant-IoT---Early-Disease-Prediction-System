"""
Retrain Kidney Disease Model for Production
===========================================
Objective: Train a robust model on the raw dataset with proper feature tracking.
Focus: Weighted Metrics (F1) to handle class imbalance.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Ensure output directories exist
os.makedirs('models/kidney_classifiers', exist_ok=True)

def train_production_model():
    print("="*70)
    print("🔄 STARTING PRODUCTION RETRAINING")
    print("="*70)

    # 1. Load Data
    data_path = 'data/raw/kidney_disease_dataset.csv'
    print(f"📖 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Define Features
    target_col = 'Target'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"   Features: {len(X.columns)}")
    print(f"   Numeric: {len(numeric_features)}")
    print(f"   Categorical: {len(categorical_features)}")

    # 3. Preprocessing Pipeline
    # Numeric: Impute (median) -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute (constant) -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training Set: {len(X_train)} samples")
    print(f"   Test Set: {len(X_test)} samples")

    # 5. Encode Target
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Calculate Class Weights
    classes = np.unique(y_train_encoded)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)
    class_weights_dict = dict(zip(classes, weights))
    print(f"   Class Weights: {class_weights_dict}")

    # 6. Train Model (MLPClassifier - as recommended in reports)
    # Note: MLPClassifier doesn't accept class_weight directly in fit, handling via sample_weight is possible but rare.
    # Alternatives: RandomForest or XGBoost with weights.
    # However, report said MLP was best. Let's try MLP but we can't easily weight it in sklearn without partial_fit or careful handling.
    # Wait, simple workaround: Use RandomForest with class_weight='balanced' OR use XGBoost.
    # The report said "Neural Network (SMOTE) - Best overall".
    # Let's try RandomForest for robustness and interpretability first, with class_weight='balanced_subsample'.
    
    print("🏃 Training Random Forest (Weighted)...")
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit pipeline
    # We need to fit preprocessor first to get feature names later
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    clf.fit(X_train_processed, y_train_encoded)

    # 7. Evaluate
    print("📊 Evaluation on PROCESSED Test Set:")
    y_pred = clf.predict(X_test_processed)
    
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    print(f"   Weighted F1 Score: {f1:.4f}")

    # 8. Save Artifacts for Inference
    print("💾 Saving artifacts...")
    
    # Save Model
    joblib.dump(clf, 'models/kidney_classifiers/production_kidney_model.pkl')
    
    # Save Preprocessor (contains scaler and encoder)
    joblib.dump(preprocessor, 'models/kidney_classifiers/production_preprocessor.pkl')
    
    # Save Label Encoder
    joblib.dump(le, 'models/kidney_classifiers/production_label_encoder.pkl')
    
    # Save Metadata with Feature Names
    # For ColumnTransformer, deriving feature names can be tricky depending on version,
    # but get_feature_names_out is standard in recent sklearn.
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        feature_names_list = list(feature_names_out)
    except Exception as e:
        print(f"Warning: Could not get output feature names ({e}). Using raw lists.")
        feature_names_list = [] # This is for output features. We need INPUT features for mapping.

    metadata = {
        'model_name': 'RandomForest_Weighted',
        'input_features': {
            'numeric': numeric_features,
            'categorical': categorical_features
        },
        'output_classes': list(le.classes_),
        'performance': {'weighted_f1': f1}
    }
    
    with open('models/kidney_classifiers/production_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("✅ Training Complete and Artifacts Saved!")

if __name__ == "__main__":
    train_production_model()
