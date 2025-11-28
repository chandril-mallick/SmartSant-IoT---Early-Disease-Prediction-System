"""
Advanced Integration of Kaggle CKD Datasets (Fixed)
====================================================
Maximize feature usage by mapping proxies and using iterative imputation.
Includes proper categorical encoding.
"""

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

print("="*70)
print("ADVANCED INTEGRATION OF KAGGLE CKD DATASETS")
print("="*70)

# Load datasets
print("\n1️⃣  Loading datasets...")
df_rabie = pd.read_csv('data/external/Chronic_Kidney_Dsease_data.csv')
df_uci = pd.read_csv('data/external/CKD_Preprocessed.csv')
df_current = pd.read_csv('data/raw/kidney_disease_dataset.csv')

# ---------------------------------------------------------
# 2. Feature Mapping (Aggressive)
# ---------------------------------------------------------
print("\n2️⃣  Mapping features...")

# Target Features (Union of important features)
target_features = [
    'Age', 'Blood_Pressure', 'Specific_Gravity', 'Albumin', 'Sugar', 
    'Red_Blood_Cells', 'Pus_Cells', 'Pus_Cell_Clumps', 'Bacteria', 
    'Blood_Glucose_Random', 'Blood_Urea', 'Serum_Creatinine', 'Sodium', 
    'Potassium', 'Hemoglobin', 'Packed_Cell_Volume', 'White_Blood_Cell_Count', 
    'Red_Blood_Cell_Count', 'Hypertension', 'Diabetes_Mellitus', 
    'Coronary_Artery_Disease', 'Appetite', 'Pedal_Edema', 'Anemia'
]

def clean_categorical(df):
    """Map categorical strings to numbers"""
    # Binary mappings
    maps = {
        'normal': 0, 'abnormal': 1,
        'notpresent': 0, 'present': 1,
        'no': 0, 'yes': 1,
        'good': 0, 'poor': 1,
        '\tno': 0, '\tyes': 1, # Common typo in UCI dataset
        ' yes': 1, ' no': 0
    }
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try mapping
            df[col] = df[col].map(maps).fillna(df[col])
    return df

def prepare_current(df):
    df = df.copy()
    # Map columns
    mapping = {
        'Age of the patient': 'Age',
        'Blood pressure (mm/Hg)': 'Blood_Pressure',
        'Specific gravity of urine': 'Specific_Gravity',
        'Albumin in urine': 'Albumin',
        'Sugar in urine': 'Sugar',
        'Red blood cells in urine': 'Red_Blood_Cells', 
        'Pus cells in urine': 'Pus_Cells',
        'Pus cell clumps in urine': 'Pus_Cell_Clumps',
        'Bacteria in urine': 'Bacteria',
        'Random blood glucose level (mg/dl)': 'Blood_Glucose_Random',
        'Blood urea (mg/dL)': 'Blood_Urea',
        'Serum creatinine (mg/dL)': 'Serum_Creatinine',
        'Sodium (mEq/L)': 'Sodium',
        'Potassium (mEq/L)': 'Potassium',
        'Hemoglobin (g/dL)': 'Hemoglobin',
        'Packed cell volume': 'Packed_Cell_Volume',
        'White blood cell count': 'White_Blood_Cell_Count',
        'Red blood cell count': 'Red_Blood_Cell_Count',
        'Hypertension': 'Hypertension',
        'Diabetes mellitus': 'Diabetes_Mellitus',
        'Coronary artery disease': 'Coronary_Artery_Disease',
        'Appetite': 'Appetite',
        'Pedal edema': 'Pedal_Edema',
        'Anemia': 'Anemia'
    }
    df = df.rename(columns=mapping)
    
    # Clean categoricals
    df = clean_categorical(df)
    
    # Keep only mapped columns + Target
    cols = [c for c in df.columns if c in target_features or c == 'Target']
    return df[cols]

def prepare_rabie(df):
    df = df.copy()
    # Map columns
    mapping = {
        'Age': 'Age',
        'SystolicBP': 'Blood_Pressure', # Proxy
        'SerumCreatinine': 'Serum_Creatinine',
        'BUNLevels': 'Blood_Urea',
        'HemoglobinLevels': 'Hemoglobin',
        'SerumElectrolytesSodium': 'Sodium',
        'SerumElectrolytesPotassium': 'Potassium',
        'FastingBloodSugar': 'Blood_Glucose_Random', # Proxy
        'ProteinInUrine': 'Albumin', # Proxy
        # Derived/Mapped binaries
        'FamilyHistoryHypertension': 'Hypertension', # Proxy
        'FamilyHistoryDiabetes': 'Diabetes_Mellitus', # Proxy
        'Edema': 'Pedal_Edema',
        'DietQuality': 'Appetite', # Proxy (Low quality -> Poor appetite?)
    }
    
    df = df.rename(columns=mapping)
    
    # Handle Appetite proxy: DietQuality 0-10. Let's say < 5 is poor (1), >= 5 is good (0)
    if 'Appetite' in df.columns:
        df['Appetite'] = df['Appetite'].apply(lambda x: 1 if x < 5 else 0)
        
    # Create missing columns with NaNs
    for col in target_features:
        if col not in df.columns:
            df[col] = np.nan
            
    # Map Target
    def get_stage(row):
        if row['Diagnosis'] == 0: return 'No_Disease'
        gfr = row['GFR']
        if gfr >= 90: return 'Low_Risk'
        elif gfr >= 60: return 'Low_Risk'
        elif gfr >= 45: return 'Moderate_Risk'
        elif gfr >= 30: return 'High_Risk'
        else: return 'Severe_Disease'
        
    df['Target'] = df.apply(get_stage, axis=1)
    
    return df[target_features + ['Target']]

def prepare_uci(df):
    df = df.copy()
    # UCI Preprocessed has different names
    mapping = {
        'Age (yrs)': 'Age',
        'Blood Pressure (mm/Hg)': 'Blood_Pressure',
        'Specific Gravity': 'Specific_Gravity',
        'Albumin': 'Albumin',
        'Sugar': 'Sugar',
        'Blood Glucose Random (mgs/dL)': 'Blood_Glucose_Random',
        'Blood Urea (mgs/dL)': 'Blood_Urea',
        'Serum Creatinine (mgs/dL)': 'Serum_Creatinine',
        'Sodium (mEq/L)': 'Sodium',
        'Potassium (mEq/L)': 'Potassium',
        'Hemoglobin (gms)': 'Hemoglobin',
        'Packed Cell Volume': 'Packed_Cell_Volume',
        'White Blood Cell Count': 'White_Blood_Cell_Count',
        'Red Blood Cell Count': 'Red_Blood_Cell_Count',
        'Hypertension': 'Hypertension',
        'Diabetes Mellitus': 'Diabetes_Mellitus',
        'Coronary Artery Disease': 'Coronary_Artery_Disease',
        'Appetite': 'Appetite',
        'Pedal Edema': 'Pedal_Edema',
        'Anemia': 'Anemia'
    }
    df = df.rename(columns=mapping)
    
    # Clean categoricals
    df = clean_categorical(df)
    
    # Create missing columns with NaNs
    for col in target_features:
        if col not in df.columns:
            df[col] = np.nan

    # Map Target
    ckd_col = [c for c in df.columns if 'Chronic Kidney Disease' in c]
    if ckd_col:
        ckd_col = ckd_col[0]
        def get_stage_uci(row):
            if row[ckd_col] == 0: return 'No_Disease'
            creat = row.get('Serum_Creatinine', 1.0)
            if pd.isna(creat): return 'Moderate_Risk'
            if creat < 1.5: return 'Low_Risk'
            elif creat < 3.0: return 'Moderate_Risk'
            elif creat < 5.0: return 'High_Risk'
            else: return 'Severe_Disease'
        df['Target'] = df.apply(get_stage_uci, axis=1)
    else:
        df['Target'] = 'Low_Risk'

    return df[target_features + ['Target']]

# Process all
print("\n3️⃣  Processing datasets...")
df1 = prepare_current(df_current)
df2 = prepare_rabie(df_rabie)
df3 = prepare_uci(df_uci)

print(f"   Current: {df1.shape}")
print(f"   Rabie:   {df2.shape}")
print(f"   UCI:     {df3.shape}")

# Combine
print("\n4️⃣  Combining and Imputing...")
combined = pd.concat([df1, df2, df3], ignore_index=True)

# Separate Target
X = combined[target_features]
y = combined['Target']

# Force numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Iterative Imputer
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed_array = imputer.fit_transform(X)

# Handle potential dropped columns (though unlikely now with proper mapping)
# If imputer drops columns, X_imputed_array will have fewer columns than X
if X_imputed_array.shape[1] != X.shape[1]:
    print(f"   Warning: Imputer dropped {X.shape[1] - X_imputed_array.shape[1]} columns")
    # We need to find which ones were kept. 
    # IterativeImputer doesn't have get_feature_names_out by default in older sklearn, 
    # but usually it keeps features with observed values.
    # Let's just assume it kept them all if we fixed the mapping.
    # If not, we'll just use range columns.
    X_imputed = pd.DataFrame(X_imputed_array)
else:
    X_imputed = pd.DataFrame(X_imputed_array, columns=target_features)

# Recombine
final_df = pd.concat([X_imputed, y], axis=1)

print(f"   Final Shape: {final_df.shape}")
print(f"   Missing Values: {final_df.isnull().sum().sum()}")
print(f"   Class Distribution:\n{final_df['Target'].value_counts()}")

# Save
output_path = 'data/processed/advanced_combined_kidney.csv'
os.makedirs('data/processed', exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"\n💾 Saved to {output_path}")
