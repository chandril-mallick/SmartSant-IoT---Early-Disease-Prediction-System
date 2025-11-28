"""
Integrate Kaggle CKD Datasets with Current Data
================================================
Map features intelligently and create unified dataset
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

print("="*70)
print("INTEGRATING KAGGLE CKD DATASETS")
print("="*70)

# Load all datasets
print("\n1️⃣  Loading datasets...")
df_rabie = pd.read_csv('data/external/Chronic_Kidney_Dsease_data.csv')
df_uci = pd.read_csv('data/external/CKD_Preprocessed.csv')
df_current = pd.read_csv('data/raw/kidney_disease_dataset.csv')

print(f"   Rabie:   {df_rabie.shape}")
print(f"   UCI:     {df_uci.shape}")
print(f"   Current: {df_current.shape}")

# Feature mapping - map similar features across datasets
print("\n2️⃣  Creating feature mappings...")

# Common clinical features that should exist across datasets
feature_map_rabie = {
    'Age': 'Age of the patient',
    'SystolicBP': 'Blood pressure (mm/Hg)',  # Will need to combine with DiastolicBP
    'SerumCreatinine': 'Serum creatinine (mg/dL)',
    'BUNLevels': 'Blood urea (mg/dL)',
    'GFR': 'Glomerular filtration rate (mL/min)',
    'HemoglobinLevels': 'Hemoglobin (g/dL)',
    'SerumElectrolytesSodium': 'Sodium (mEq/L)',
    'SerumElectrolytesPotassium': 'Potassium (mEq/L)',
    'FastingBloodSugar': 'Random blood glucose level (mg/dl)',
}

feature_map_uci = {
    'Age (yrs)': 'Age of the patient',
    'Blood Pressure (mm/Hg)': 'Blood pressure (mm/Hg)',
    'Specific Gravity': 'Specific gravity of urine',
    'Albumin': 'Albumin in urine',
    'Sugar': 'Sugar in urine',
    'Blood Glucose Random (mgs/dL)': 'Random blood glucose level (mg/dl)',
    'Blood Urea (mgs/dL)': 'Blood urea (mg/dL)',
    'Serum Creatinine (mgs/dL)': 'Serum creatinine (mg/dL)',
    'Sodium (mEq/L)': 'Sodium (mEq/L)',
    'Potassium (mEq/L)': 'Potassium (mEq/L)',
    'Hemoglobin (gms)': 'Hemoglobin (g/dL)',
}

# Map Rabie's CKD diagnosis to 5-class system
# Diagnosis: 0 = No CKD, 1 = CKD
# We'll use GFR to determine severity for CKD cases
print("\n3️⃣  Mapping Rabie's dataset to 5-class system...")

def map_rabie_to_5class(row):
    """Map Rabie's binary CKD + GFR to 5-class system"""
    if row['Diagnosis'] == 0:
        return 'No_Disease'
    else:
        # CKD staging based on GFR
        gfr = row['GFR']
        if gfr >= 60:
            return 'Low_Risk'  # Stage 1-2
        elif gfr >= 45:
            return 'Moderate_Risk'  # Stage 3a
        elif gfr >= 30:
            return 'High_Risk'  # Stage 3b
        else:
            return 'Severe_Disease'  # Stage 4-5

df_rabie['Target'] = df_rabie.apply(map_rabie_to_5class, axis=1)
print(f"   Rabie class distribution:")
print(df_rabie['Target'].value_counts())

# Map UCI's CKD to 5-class (they don't have GFR, so we'll use creatinine)
print("\n4️⃣  Mapping UCI dataset to 5-class system...")

# Find the CKD column in UCI
ckd_col = [col for col in df_uci.columns if 'Chronic Kidney Disease' in col or 'ckd' in col.lower()]
if ckd_col:
    ckd_col = ckd_col[0]
    print(f"   Found CKD column: {ckd_col}")
    
    def map_uci_to_5class(row):
        """Map UCI's CKD + Creatinine to 5-class"""
        if pd.isna(row.get(ckd_col, 1)) or row.get(ckd_col, 1) == 0:
            return 'No_Disease'
        else:
            # Use serum creatinine for staging (higher = worse)
            creat = row.get('Serum Creatinine (mgs/dL)', 1.0)
            if creat < 1.5:
                return 'Low_Risk'
            elif creat < 2.5:
                return 'Moderate_Risk'
            elif creat < 4.0:
                return 'High_Risk'
            else:
                return 'Severe_Disease'
    
    df_uci['Target'] = df_uci.apply(map_uci_to_5class, axis=1)
    print(f"   UCI class distribution:")
    print(df_uci['Target'].value_counts())
else:
    print("   Warning: Could not find CKD column, marking all as CKD")
    df_uci['Target'] = 'Low_Risk'  # Conservative assumption

# Extract common features
print("\n5️⃣  Extracting common features...")

# Get available mapped features from each dataset
rabie_features = {}
for rabie_col, common_col in feature_map_rabie.items():
    if rabie_col in df_rabie.columns:
        rabie_features[common_col] = df_rabie[rabie_col]

uci_features = {}
for uci_col, common_col in feature_map_uci.items():
    if uci_col in df_uci.columns:
        uci_features[common_col] = df_uci[uci_col]

# Create simplified datasets with common features
common_features = ['Age of the patient', 'Blood pressure (mm/Hg)', 'Serum creatinine (mg/dL)', 
                   'Blood urea (mg/dL)', 'Sodium (mEq/L)', 'Potassium (mEq/L)', 
                   'Hemoglobin (g/dL)', 'Random blood glucose level (mg/dl)']

# Build Rabie subset
rabie_subset = pd.DataFrame()
rabie_subset['Target'] = df_rabie['Target']
for feat in common_features:
    if feat in rabie_features:
        rabie_subset[feat] = rabie_features[feat]
    else:
        rabie_subset[feat] = np.nan

# Build UCI subset  
uci_subset = pd.DataFrame()
uci_subset['Target'] = df_uci['Target']
for feat in common_features:
    if feat in uci_features:
        uci_subset[feat] = uci_features[feat]
    else:
        uci_subset[feat] = np.nan

# Get current dataset subset
current_subset = df_current[['Target'] + [f for f in common_features if f in df_current.columns]].copy()

print(f"\n   Rabie subset: {rabie_subset.shape}")
print(f"   UCI subset: {uci_subset.shape}")
print(f"   Current subset: {current_subset.shape}")

# Combine datasets
print("\n6️⃣  Combining datasets...")
combined = pd.concat([current_subset, rabie_subset, uci_subset], ignore_index=True)

print(f"   Combined shape: {combined.shape}")
print(f"\n   Combined class distribution:")
print(combined['Target'].value_counts())
print(f"\n   Percentage:")
print(combined['Target'].value_counts(normalize=True) * 100)

# Save combined dataset
output_path = 'data/processed/combined_kidney_dataset.csv'
os.makedirs('data/processed', exist_ok=True)
combined.to_csv(output_path, index=False)

print(f"\n💾 Saved combined dataset to: {output_path}")

# Summary
print("\n" + "="*70)
print("INTEGRATION SUMMARY")
print("="*70)
print(f"\n📊 Dataset Sizes:")
print(f"   Original:  {len(df_current):,} samples")
print(f"   + Rabie:   {len(df_rabie):,} samples")
print(f"   + UCI:     {len(df_uci):,} samples")
print(f"   = Combined: {len(combined):,} samples (+{len(combined) - len(df_current):,})")

print(f"\n📈 Class Distribution Improvement:")
print(f"   Severe_Disease: {len(df_current[df_current['Target']=='Severe_Disease']):,} → {len(combined[combined['Target']=='Severe_Disease']):,} "
      f"(+{len(combined[combined['Target']=='Severe_Disease']) - len(df_current[df_current['Target']=='Severe_Disease']):,})")

print("\n✅ Integration complete!")
print("\nNext: Train models on combined dataset")
