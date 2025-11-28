"""
Fix Kidney Model Dependencies
=============================
Generate a compatible LabelEncoder for the LightGBM model
to ensure app.py works correctly.
"""

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

print("Fixing Kidney Model Dependencies...")

# Define classes in the order used by train_combined_kidney.py
# target_map = {
#     'No_Disease': 0,
#     'Low_Risk': 1,
#     'Moderate_Risk': 2,
#     'High_Risk': 3,
#     'Severe_Disease': 4
# }
classes = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'High_Risk', 'Severe_Disease']

# Create and fit LabelEncoder
le = LabelEncoder()
le.classes_ = np.array(classes)

# Verify
print(f"Classes: {le.classes_}")
print(f"Transform 'No_Disease': {le.transform(['No_Disease'])[0]} (Expected 0)")
print(f"Transform 'Severe_Disease': {le.transform(['Severe_Disease'])[0]} (Expected 4)")

# Save
save_path = 'models/kidney_classifiers/label_encoder.pkl'
joblib.dump(le, save_path)
print(f"✅ Saved LabelEncoder to {save_path}")
