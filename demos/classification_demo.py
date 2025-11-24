"""
Demonstration: Urine Disease Classification
Shows how to make predictions with the trained model.
"""

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.urine_preprocessor import preprocess_urine_data
from config import data_config

# Load the best trained model
model_path = 'models/urine_classifiers/logistic_regression.pkl'
print("\n" + "="*70)
print("URINE DISEASE CLASSIFICATION - DEMO")
print("="*70)

print(f"\n📁 Loading model from: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Load and preprocess data
print("\n📊 Loading test data...")
data_path = os.path.join(data_config.RAW_DATA_DIR, data_config.URINE_CSV)
X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
    data_path=data_path,
    target='Diagnosis',
    test_size=0.2,
    random_state=42,
    handle_imbalance=True
)

print(f"✅ Test set loaded: {len(y_test)} samples")

# Make predictions on test set
print("\n🔮 Making predictions...")
y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Show some example predictions
print("\n" + "="*70)
print("EXAMPLE PREDICTIONS")
print("="*70)

print(f"\n{'Sample':<8} {'True':<12} {'Predicted':<12} {'Prob Negative':<15} {'Prob Positive':<15}")
print("-" * 70)

# Show first 10 samples
for i in range(min(10, len(y_test_array))):
    true_label = 'POSITIVE' if y_test_array[i] == 1 else 'NEGATIVE'
    pred_label = 'POSITIVE' if y_pred[i] == 1 else 'NEGATIVE'
    prob_neg = y_pred_proba[i][0]
    prob_pos = y_pred_proba[i][1]
    
    # Mark correct/incorrect predictions
    marker = '✅' if y_test_array[i] == y_pred[i] else '❌'
    
    print(f"{marker} {i+1:<6} {true_label:<12} {pred_label:<12} {prob_neg:<15.4f} {prob_pos:<15.4f}")

# Summary statistics
correct = (y_pred == y_test_array).sum()
total = len(y_test_array)
accuracy = correct / total

print("\n" + "="*70)
print("CLASSIFICATION SUMMARY")
print("="*70)
print(f"\nTotal samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Accuracy: {accuracy:.2%}")

# Breakdown by class
positive_samples = (y_test_array == 1).sum()
negative_samples = (y_test_array == 0).sum()

true_positives = ((y_test_array == 1) & (y_pred == 1)).sum()
true_negatives = ((y_test_array == 0) & (y_pred == 0)).sum()
false_positives = ((y_test_array == 0) & (y_pred == 1)).sum()
false_negatives = ((y_test_array == 1) & (y_pred == 0)).sum()

print(f"\n📊 Detailed Breakdown:")
print(f"   Positive cases in test: {positive_samples} ({positive_samples/total*100:.1f}%)")
print(f"   Negative cases in test: {negative_samples} ({negative_samples/total*100:.1f}%)")
print(f"\n   True Positives:  {true_positives}/{positive_samples} ({true_positives/positive_samples*100 if positive_samples > 0 else 0:.1f}%)")
print(f"   True Negatives:  {true_negatives}/{negative_samples} ({true_negatives/negative_samples*100:.1f}%)")
print(f"   False Positives: {false_positives}/{negative_samples} ({false_positives/negative_samples*100:.1f}%)")
print(f"   False Negatives: {false_negatives}/{positive_samples} ({false_negatives/positive_samples*100 if positive_samples > 0 else 0:.1f}%)")

print("\n" + "="*70)
print("✅ CLASSIFICATION DEMO COMPLETE!")
print("="*70)
