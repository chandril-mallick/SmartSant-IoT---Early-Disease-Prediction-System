"""
Unified Medical Prediction System
Integrates urine disease (UTI) and kidney disease (CKD) classifiers.
Routes patient data to appropriate classifier(s) automatically.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Optional, List, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config


class UnifiedMedicalPredictor:
    """
    Unified interface for urine and kidney disease prediction.
    Automatically routes to appropriate classifier based on available data.
    """
    
    def __init__(self):
        """Initialize both classifiers."""
        self.urine_model = None
        self.kidney_model = None
        self.kidney_label_encoder = None
        
        # Load models
        self._load_urine_classifier()
        self._load_kidney_classifier()
    
    def _load_urine_classifier(self):
        """Load urine disease (UTI) classifier."""
        try:
            urine_dir = os.path.join(data_config.MODELS_DIR, 'urine_classifiers')
            
            # Try loading optimized model first
            optimized_model_path = os.path.join(urine_dir, 'optimized_urine_classifier.pkl')
            optimized_metadata_path = os.path.join(urine_dir, 'optimized_model_metadata.json')
            
            if os.path.exists(optimized_model_path) and os.path.exists(optimized_metadata_path):
                self.urine_model = joblib.load(optimized_model_path)
                with open(optimized_metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.urine_model_name = f"Optimized {metadata.get('model_name', 'Model')}"
                
                # Get performance metric (handle different metadata formats)
                if 'performance' in metadata:
                    perf = metadata['performance']
                    metric = f"Accuracy: {perf.get('accuracy', 0):.2%}"
                else:
                    metric = "Performance data unavailable"
                    
                print(f"✅ Loaded Urine Classifier: {self.urine_model_name}")
                print(f"   {metric}")
                
            else:
                # Fallback to standard best model
                metadata_path = os.path.join(urine_dir, 'best_model_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_path = os.path.join(urine_dir, metadata['best_model_file'])
                    self.urine_model = joblib.load(model_path)
                    self.urine_model_name = metadata['best_model']
                    print(f"✅ Loaded Urine Classifier: {self.urine_model_name}")
                else:
                    print("⚠️  No urine classifier found")

        except Exception as e:
            print(f"⚠️  Could not load urine classifier: {e}")
    
    def _load_kidney_classifier(self):
        """Load kidney disease (CKD) classifier."""
        try:
            kidney_dir = os.path.join(data_config.MODELS_DIR, 'kidney_classifiers')
            
            # Try loading optimized model first
            optimized_model_path = os.path.join(kidney_dir, 'optimized_kidney_classifier.pkl')
            optimized_metadata_path = os.path.join(kidney_dir, 'optimized_model_metadata.json')
            
            if os.path.exists(optimized_model_path) and os.path.exists(optimized_metadata_path):
                self.kidney_model = joblib.load(optimized_model_path)
                with open(optimized_metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.kidney_model_name = f"Optimized {metadata.get('model_name', 'Model')}"
                
                # Load label encoder
                le_path = os.path.join(kidney_dir, 'label_encoder.pkl')
                if os.path.exists(le_path):
                    self.kidney_label_encoder = joblib.load(le_path)
                
                # Get performance metric
                if 'performance' in metadata:
                    perf = metadata['performance']
                    metric = f"CV F1: {perf.get('cv_f1', 0):.2%}, Test Acc: {perf.get('accuracy', 0):.2%}"
                else:
                    metric = "Performance data unavailable"
                
                print(f"✅ Loaded Kidney Classifier: {self.kidney_model_name}")
                print(f"   {metric}")
                
            else:
                # Fallback to standard best model
                metadata_path = os.path.join(kidney_dir, 'best_model_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_path = os.path.join(kidney_dir, metadata['best_model_file'])
                    self.kidney_model = joblib.load(model_path)
                    self.kidney_model_name = metadata['best_model']
                    
                    # Load label encoder
                    le_path = os.path.join(kidney_dir, 'label_encoder.pkl')
                    if os.path.exists(le_path):
                        self.kidney_label_encoder = joblib.load(le_path)
                        
                    print(f"✅ Loaded Kidney Classifier: {self.kidney_model_name}")
                else:
                    print("⚠️  No kidney classifier found")
            
        except Exception as e:
            print(f"⚠️  Could not load kidney classifier: {e}")
    
    def predict_urine_disease(self, patient_data: Dict) -> Dict:
        """
        Predict UTI from urine test data.
        
        Args:
            patient_data: Dict with urine test parameters
            
        Returns:
            Prediction results
        """
        if self.urine_model is None:
            return {'error': 'Urine classifier not loaded'}
        
        # Note: This is simplified - in production, you'd need to preprocess the data
        # using the same UrinePreprocessor that was used during training
        
        try:
            # For demo: assuming data is already preprocessed
            # In production: df = preprocess_patient_data(patient_data)
            
            return {
                'classifier': 'Urine Disease (UTI)',
                'model': self.urine_model_name,
                'status': 'Preprocessing required',
                'note': 'Full preprocessing pipeline integration needed for production'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_kidney_disease(self, patient_data: Dict) -> Dict:
        """
        Predict CKD risk from kidney disease markers.
        
        Args:
            patient_data: Dict with kidney function parameters
            
        Returns:
            Prediction results
        """
        if self.kidney_model is None:
            return {'error': 'Kidney classifier not loaded'}
        
        try:
            # Note: This is simplified - in production, you'd need to preprocess the data
            # using the same KidneyPreprocessor that was used during training
            
            return {
                'classifier': 'Kidney Disease (CKD)',
                'model': self.kidney_model_name,
                'classes': list(self.kidney_label_encoder.classes_),
                'status': 'Preprocessing required',
                'note': 'Full preprocessing pipeline integration needed for production'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_all(self, patient_data: Dict) -> Dict:
        """
        Run both classifiers if sufficient data is available.
        
        Args:
            patient_data: Dict with all available patient data
            
        Returns:
            Combined predictions
        """
        results = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'predictions': {}
        }
        
        # Check for urine test data
        urine_fields = ['WBC', 'RBC', 'Bacteria', 'pH', 'Protein']
        has_urine_data = any(field in patient_data for field in urine_fields)
        
        if has_urine_data:
            results['predictions']['urine'] = self.predict_urine_disease(patient_data)
        
        # Check for kidney function data
        kidney_fields = ['Serum creatinine (mg/dl)', 'Blood urea (mg/dl)', 'Estimated Glomerular Filtration Rate (eGFR)']
        has_kidney_data = any(field in patient_data for field in kidney_fields)
        
        if has_kidney_data:
            results['predictions']['kidney'] = self.predict_kidney_disease(patient_data)
        
        if not has_urine_data and not has_kidney_data:
            results['error'] = 'Insufficient data for classification'
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'urine_classifier': {
                'loaded': self.urine_model is not None,
                'model': self.urine_model_name if self.urine_model else None,
                'type': 'Binary (UTI: Positive/Negative)',
                'features': 'Urine test parameters (WBC, RBC, pH, etc.)'
            },
            'kidney_classifier': {
                'loaded': self.kidney_model is not None,
                'model': self.kidney_model_name if self.kidney_model else None,
                'type': 'Multi-class (5 CKD risk levels)',
                'classes': list(self.kidney_label_encoder.classes_) if self.kidney_label_encoder else None,
                'features': 'Blood tests, urine markers, clinical indicators'
            }
        }


def demo_unified_predictor():
    """Demonstrate the unified prediction system."""
    print("\n" + "="*70)
    print("UNIFIED MEDICAL PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize predictor
    predictor = UnifiedMedicalPredictor()
    
    # Show model info
    print("\n📊 Model Information:")
    info = predictor.get_model_info()
    
    print(f"\n🔬 Urine Disease Classifier:")
    print(f"  Status: {'✅ Loaded' if info['urine_classifier']['loaded'] else '❌ Not loaded'}")
    if info['urine_classifier']['loaded']:
        print(f"  Model: {info['urine_classifier']['model']}")
        print(f"  Type: {info['urine_classifier']['type']}")
    
    print(f"\n🩺 Kidney Disease Classifier:")
    print(f"  Status: {'✅ Loaded' if info['kidney_classifier']['loaded'] else '❌ Not loaded'}")
    if info['kidney_classifier']['loaded']:
        print(f"  Model: {info['kidney_classifier']['model']}")
        print(f"  Type: {info['kidney_classifier']['type']}")
        print(f"  Classes: {', '.join(info['kidney_classifier']['classes'])}")
    
    # Example patient data
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Patient with urine data
    print("\n📋 Patient 1: Urine Test Data")
    patient1 = {
        'patient_id': 'P001',
        'WBC': '10-15',
        'RBC': '0-2',
        'Bacteria': 'MODERATE',
        'pH': 6.0,
        'Protein': 'TRACE'
    }
    result1 = predictor.predict_all(patient1)
    print(json.dumps(result1, indent=2))
    
    # Patient with kidney data
    print("\n📋 Patient 2: Kidney Function Data")
    patient2 = {
        'patient_id': 'P002',
        'Serum creatinine (mg/dl)': 2.5,
        'Blood urea (mg/dl)': 80.0,
        'Estimated Glomerular Filtration Rate (eGFR)': 45.0
    }
    result2 = predictor.predict_all(patient2)
    print(json.dumps(result2, indent=2))
    
    # Patient with both
    print("\n📋 Patient 3: Complete Data (Both Tests)")
    patient3 = {
        'patient_id': 'P003',
        'WBC': '5-8',
        'RBC': '0-1',
        'pH': 6.5,
        'Serum creatinine (mg/dl)': 1.2,
        'Estimated Glomerular Filtration Rate (eGFR)': 85.0
    }
    result3 = predictor.predict_all(patient3)
    print(json.dumps(result3, indent=2))
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\n💡 Note: Full preprocessing integration required for production use")
    print("   Current version demonstrates model loading and routing logic")


if __name__ == "__main__":
    demo_unified_predictor()
