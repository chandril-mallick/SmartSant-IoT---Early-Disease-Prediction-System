"""
Urine Disease Classification - Multi-Model Training Pipeline
Trains and compares multiple ML models for UTI prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.urine_preprocessor import preprocess_urine_data
from config import data_config

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False


class UrineClassifierPipeline:
    """Complete training and evaluation pipeline for urine disease classification."""
    
    def __init__(self, save_dir=None):
        """
        Initialize the pipeline.
        
        Args:
            save_dir: Directory to save trained models
        """
        self.save_dir = save_dir or os.path.join(data_config.MODELS_DIR, 'urine_classifiers')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def create_models(self):
        """Create all models to train."""
        print("\n" + "="*70)
        print("CREATING CLASSIFICATION MODELS")
        print("="*70)
        
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=15,  # For class imbalance (1355/81 ≈ 17)
                random_state=42,
                eval_metric='logloss'
            )
        
        self.models = models
        
        for name in models.keys():
            print(f"  ✅ {name}")
        
        print(f"\nTotal models: {len(models)}")
        return models
    
    def train_model(self, name, model, X_train, y_train, X_test, y_test):
        """
        Train a single model and evaluate it.
        
        Returns:
            Dictionary with training results
        """
        print(f"\n{'-'*70}")
        print(f"Training: {name}")
        print(f"{'-'*70}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba_train = y_pred_train
            y_pred_proba_test = y_pred_test
        
        # Compute metrics
        results = {
            'model_name': name,
            'training_time': training_time,
            
            # Training metrics
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_pred_proba_train),
            
            # Test metrics
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"\n  Training Set:")
        print(f"    Accuracy:  {results['train_accuracy']:.4f}")
        print(f"    Precision: {results['train_precision']:.4f}")
        print(f"    Recall:    {results['train_recall']:.4f}")
        print(f"    F1-Score:  {results['train_f1']:.4f}")
        print(f"    ROC-AUC:   {results['train_roc_auc']:.4f}")
        
        print(f"\n  Test Set:")
        print(f"    Accuracy:  {results['test_accuracy']:.4f}")
        print(f"    Precision: {results['test_precision']:.4f}")
        print(f"    Recall:    {results['test_recall']:.4f}")
        print(f"    F1-Score:  {results['test_f1']:.4f}")
        print(f"    ROC-AUC:   {results['test_roc_auc']:.4f}")
        
        # Check for overfitting
        overfit_score = results['train_accuracy'] - results['test_accuracy']
        if overfit_score > 0.1:
            print(f"\n  ⚠️  Warning: Potential overfitting detected ({overfit_score:.2%})")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and collect results."""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            results = self.train_model(name, model, X_train, y_train, X_test, y_test)
            self.results[name] = results
        
        return self.results
    
    def compare_models(self):
        """Compare all trained models and select the best."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1-Score': results['test_f1'],
                'ROC-AUC': results['test_roc_auc'],
                'Time (s)': results['training_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        print("\n" + df.to_string(index=False))
        
        # Select best model (based on F1-score for imbalanced data)
        best_idx = df['F1-Score'].idxmax()
        self.best_model_name = df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   F1-Score: {df.loc[best_idx, 'F1-Score']:.4f}")
        print(f"   Recall: {df.loc[best_idx, 'Recall']:.4f}")
        print(f"   ROC-AUC: {df.loc[best_idx, 'ROC-AUC']:.4f}")
        
        return df
    
    def save_models(self):
        """Save all trained models."""
        print(f"\n💾 Saving models to: {self.save_dir}")
        
        for name, model in self.models.items():
            model_filename = name.lower().replace(' ', '_') + '.pkl'
            model_path = os.path.join(self.save_dir, model_filename)
            joblib.dump(model, model_path)
            print(f"  ✅ Saved: {model_filename}")
        
        # Save results
        results_path = os.path.join(self.save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✅ Saved: training_results.json")
        
        # Save best model metadata
        metadata = {
            'best_model': self.best_model_name,
            'best_model_file': self.best_model_name.lower().replace(' ', '_') + '.pkl',
            'performance': self.results[self.best_model_name]
        }
        metadata_path = os.path.join(self.save_dir, 'best_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved: best_model_metadata.json")
    
    def run_full_pipeline(self, data_path=None):
        """Run the complete training pipeline."""
        print("\n" + "="*70)
        print("URINE DISEASE CLASSIFICATION - TRAINING PIPELINE")
        print("="*70)
        
        # Load and preprocess data
        print("\n📂 Loading and preprocessing data...")
        if data_path is None:
            data_path = os.path.join(data_config.RAW_DATA_DIR, data_config.URINE_CSV)
        
        X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
            data_path=data_path,
            target='Diagnosis',
            test_size=0.2,
            random_state=42,
            handle_imbalance=True
        )
        
        print(f"  Training samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Positive cases (train): {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Positive cases (test): {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
        
        # Create models
        self.create_models()
        
        # Train all models
        self.train_all_models(X_train, y_train, X_test, y_test)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Save everything
        self.save_models()
        
        # Save comparison table
        comparison_path = os.path.join(self.save_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✅ Saved: model_comparison.csv")
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n📁 All models saved to: {self.save_dir}")
        print(f"🏆 Best model: {self.best_model_name}")
        
        return self.results


if __name__ == "__main__":
    # Run the pipeline
    pipeline = UrineClassifierPipeline()
    results = pipeline.run_full_pipeline()
    
    print("\n🎉 Ready to make predictions!")
    print(f"   Load best model from: {pipeline.save_dir}")
