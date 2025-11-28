"""
Explore and Integrate Kaggle CKD Datasets
==========================================
Analyze the two downloaded datasets and prepare for integration
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("EXPLORING KAGGLE CKD DATASETS")
print("="*70)

# Load Dataset 1: Rabie's Comprehensive CKD Dataset
print("\n1️⃣  Loading Rabie's Comprehensive CKD Dataset...")
rabie_path = 'data/external/Chronic_Kidney_Dsease_data.csv'
df_rabie = pd.read_csv(rabie_path)

print(f"   Shape: {df_rabie.shape}")
print(f"   Columns ({len(df_rabie.columns)}): {list(df_rabie.columns[:10])}...")
print(f"\n   First few rows:")
print(df_rabie.head())

print(f"\n   Data types:")
print(df_rabie.dtypes.value_counts())

print(f"\n   Missing values:")
missing = df_rabie.isnull().sum()
print(missing[missing > 0])

print(f"\n   Target variable distribution:")
if 'Diagnosis' in df_rabie.columns:
    print(df_rabie['Diagnosis'].value_counts())
elif 'diagnosis' in df_rabie.columns:
    print(df_rabie['diagnosis'].value_counts())
else:
    print("   Checking all columns for target...")
    for col in df_rabie.columns:
        if 'target' in col.lower() or 'class' in col.lower() or 'ckd' in col.lower():
            print(f"   Found: {col}")
            print(df_rabie[col].value_counts())

# Load Dataset 2: Preprocessed UCI CKD Dataset
print("\n" + "="*70)
print("2️⃣  Loading Preprocessed UCI CKD Dataset...")
uci_path = 'data/external/CKD_Preprocessed.csv'
df_uci = pd.read_csv(uci_path)

print(f"   Shape: {df_uci.shape}")
print(f"   Columns ({len(df_uci.columns)}): {list(df_uci.columns[:10])}...")
print(f"\n   First few rows:")
print(df_uci.head())

print(f"\n   Data types:")
print(df_uci.dtypes.value_counts())

print(f"\n   Missing values:")
missing_uci = df_uci.isnull().sum()
print(missing_uci[missing_uci > 0] if missing_uci.sum() > 0 else "   None!")

print(f"\n   Target variable distribution:")
if 'classification' in df_uci.columns:
    print(df_uci['classification'].value_counts())
elif 'class' in df_uci.columns:
    print(df_uci['class'].value_counts())
else:
    print("   Checking all columns for target...")
    for col in df_uci.columns:
        if 'target' in col.lower() or 'class' in col.lower() or 'ckd' in col.lower():
            print(f"   Found: {col}")
            print(df_uci[col].value_counts())

# Load Current Dataset
print("\n" + "="*70)
print("3️⃣  Loading Current Kidney Disease Dataset...")
current_path = 'data/raw/kidney_disease_dataset.csv'
df_current = pd.read_csv(current_path)

print(f"   Shape: {df_current.shape}")
print(f"   Columns ({len(df_current.columns)}): {list(df_current.columns[:10])}...")

print(f"\n   Target variable distribution:")
if 'Target' in df_current.columns:
    print(df_current['Target'].value_counts())
    print(f"\n   Percentage:")
    print(df_current['Target'].value_counts(normalize=True) * 100)

# Feature Overlap Analysis
print("\n" + "="*70)
print("4️⃣  Feature Overlap Analysis")
print("="*70)

rabie_cols = set(df_rabie.columns)
uci_cols = set(df_uci.columns)
current_cols = set(df_current.columns)

print(f"\n   Rabie's dataset: {len(rabie_cols)} features")
print(f"   UCI dataset: {len(uci_cols)} features")
print(f"   Current dataset: {len(current_cols)} features")

# Common features
common_all = rabie_cols & uci_cols & current_cols
print(f"\n   Common to all 3: {len(common_all)}")
if common_all:
    print(f"   {sorted(common_all)}")

common_rabie_current = rabie_cols & current_cols
print(f"\n   Common to Rabie + Current: {len(common_rabie_current)}")
if len(common_rabie_current) > 0 and len(common_rabie_current) < 20:
    print(f"   {sorted(common_rabie_current)}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n📊 Dataset Sizes:")
print(f"   Current:  {len(df_current):,} samples")
print(f"   Rabie:    {len(df_rabie):,} samples")
print(f"   UCI:      {len(df_uci):,} samples")
print(f"   Combined: {len(df_current) + len(df_rabie) + len(df_uci):,} samples")

print(f"\n📈 Expected Improvement:")
print(f"   Current Macro F1: 19.69%")
print(f"   Target Macro F1:  65-80%")
print(f"   Improvement:      +3-4x")

print("\n✅ Data exploration complete!")
print("\nNext steps:")
print("1. Map CKD classes to 5-class system")
print("2. Align features across datasets")
print("3. Merge datasets")
print("4. Train improved models")
