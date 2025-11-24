"""
Urine Disease Prediction Interface
Easy-to-use prediction system for UTI diagnosis from urine test results.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Union, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config
from preprocessing.urine_preprocessor import UrinePreprocessor


class UrineDiseasePredictor:
    """
    User-friendly interface for predicting UTI from urine test results.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model file. If None, loads best model.
        """
        self.models_dir = os.path.join(data_config.MODELS_DIR, 'urine_classifiers')
        
        # Load best model metadata
        metadata_path = os.path.join(self.models_dir, 'best_model_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No trained models found at {self.models_dir}. "
                "Please run train_urine_classifiers.py first."
            )
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model
        if model_path is None:
            model_path = os.path.join(self.models_dir, self.metadata['best_model_file'])
        
        self.model = joblib.load(model_path)
        self.model_name = self.metadata['best_model']
        
        print(f"✅ Loaded model: {self.model_name}")
        print(f"   Performance (Test Set):")
        print(f"   - Recall: {self.metadata['performance']['test_recall']:.2%}")
        print(f"   - Precision: {self.metadata['performance']['test_precision']:.2%}")
        print(f"   - ROC-AUC: {self.metadata['performance']['test_roc_auc']:.4f}")
        
        # Load preprocessor (we'll create a new one for prediction)
        self.preprocessor = None
    
    def _prepare_input(self, patient_data: Dict) -> np.ndarray:
        """
        Prepare patient data for prediction.
        
        Args:
            patient_data: Dictionary with test results
            
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # For now, we'll need the preprocessor from training
        # In production, save the preprocessor with the model
        print("⚠️  Note: Using simplified preprocessing for prediction")
        print("   For production, save and load the full preprocessor")
        
        # Basic preprocessing (you should save/load the actual preprocessor)
        # This is a simplified version
        return df.values
    
    def predict(
        self,
        patient_data: Dict,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Make a prediction for a patient.
        
        Args:
            patient_data: Dictionary with urine test results
                Example:
                {
                    'Age': 25,
                    'Gender': 'FEMALE',
                    'pH': 6.0,
                    'WBC': '10-15',
                    'RBC': '0-2',
                    'Protein': 'TRACE',
                    ...
                }
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare input
        X = self._prepare_input(patient_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        result = {
            'diagnosis': 'POSITIVE' if prediction == 1 else 'NEGATIVE',
            'prediction_code': int(prediction)
        }
        
        # Add probabilities if available
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            result['probability_negative'] = float(proba[0])
            result['probability_positive'] = float(proba[1])
            result['confidence'] = float(max(proba))
            
            # Add confidence level
            if result['confidence'] > 0.9:
                result['confidence_level'] = 'VERY HIGH'
            elif result['confidence'] > 0.75:
                result['confidence_level'] = 'HIGH'
            elif result['confidence'] > 0.6:
                result['confidence_level'] = 'MEDIUM'
            else:
                result['confidence_level'] = 'LOW'
        
        return result
    
    def predict_batch(
        self,
        patients_data: List[Dict]
    ) -> List[Dict]:
        """
        Make predictions for multiple patients.
        
        Args:
            patients_data: List of patient data dictionaries
            
        Returns:
            List of prediction results
        """
        return [self.predict(patient) for patient in patients_data]
    
    def explain_prediction(self, patient_data: Dict) -> Dict:
        """
        Explain which features contributed most to the prediction.
        
        Args:
            patient_data: Dictionary with urine test results
            
        Returns:
            Dictionary with explanation
        """
        result = self.predict(patient_data)
        
        # Add feature importance if available
        if hasattr(self.model, 'coef_'):
            # For Logistic Regression
            feature_importance = self.model.coef_[0]
            result['feature_importance'] = {
                'available': True,
                'note': 'Positive values increase UTI risk, negative values decrease it'
            }
        elif hasattr(self.model, 'feature_importances_'):
            # For Random Forest
            feature_importance = self.model.feature_importances_
            result['feature_importance'] = {
                'available': True,
                'note': 'Higher values = more important for prediction'
            }
        else:
            result['feature_importance'] = {
                'available': False,
                'note': 'Feature importance not available for this model type'
            }
        
        return result


# Simple command-line interface
def interactive_prediction():
    """Interactive command-line interface for predictions."""
    print("\n" + "="*70)
    print("URINE DISEASE PREDICTION - INTERACTIVE MODE")
    print("="*70)
    
    # Load predictor
    try:
        predictor = UrineDiseasePredictor()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    print("\n📝 Enter patient information:")
    print("   (Press Enter to use example values)")
    
    #Example patient data for demonstration
    example_patient = {
        'Age': 25,
        'Gender': 'FEMALE',
        'Color': 'YELLOW',
        'Transparency': 'SLIGHTLY HAZY',
        'Glucose': 'NEGATIVE',
        'Protein': 'TRACE',
        'pH': 6.0,
        'Specific Gravity': 1.020,
        'WBC': '10-15',
        'RBC': '0-2',
        'Epithelial Cells': 'FEW',
        'Mucous Threads': 'RARE',
        'Amorphous Urates': 'NONE SEEN',
        'Bacteria': 'MODERATE'
    }
    
    use_example = input("\n   Use example patient? (y/n): ").lower().strip()
    
    if use_example == 'y':
        patient_data = example_patient
        print("\n   Using example patient data:")
        for key, value in patient_data.items():
            print(f"     {key}: {value}")
    else:
        print("\n   ⚠️  Custom input not yet implemented.")
        print("   Using example patient for now...")
        patient_data = example_patient
    
    # Make prediction
    print("\n🔮 Making prediction...")
    result = predictor.predict(patient_data)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\n🏥 Diagnosis: {result['diagnosis']}")
    
    if 'probability_positive' in result:
        print(f"\n📊 Probabilities:")
        print(f"   UTI Positive: {result['probability_positive']*100:.2f}%")
        print(f"   UTI Negative: {result['probability_negative']*100:.2f}%")
        print(f"\n💯 Confidence: {result['confidence']*100:.2f}% ({result['confidence_level']})")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run interactive mode
    interactive_prediction()
