# Kidney Disease Data Preprocessing - Complete Guide

## ✅ Preprocessing Complete!

Successfully implemented comprehensive preprocessing for Chronic Kidney Disease (CKD) classification dataset.

---

## 📊 Dataset Summary

### Original Data
- **Total Samples**: 20,538 patient records
- **Total Features**: 42
  - **Numeric**: 28 (age, blood tests, urine tests, clinical markers)
  - **Categorical**: 14 (yes/no, normal/abnormal, good/poor, activity levels)
- **Target**: 5-class classification
  - No_Disease: 16,432 (80.0%)
  - Low_Risk: 2,054 (10.0%)
  - Moderate_Risk: 821 (4.0%)
  - High_Risk: 821 (4.0%)
  - Severe_Disease: 410 (2.0%)
- **Missing Values**: None in original dataset
- **Data Quality**: Clean, well-structured

---

## 🔧 Preprocessing Steps Implemented

### 1. ✅ **Outlier Detection & Removal** (IQR Method)
- **Method**: Interquartile Range (IQR)
- **Threshold**: 1.5 × IQR
- **Process**:
  - Q1 = 25th percentile
  - Q3 = 75th percentile
  - Lower Bound = Q1 - 1.5 × IQR
  - Upper Bound = Q3 + 1.5 × IQR
  - Values outside bounds → Set to NaN
- **Purpose**: Remove extreme values that could skew model training

### 2. ✅ **Missing Value Imputation** (KNN)
- **Method**: K-Nearest Neighbors Imputation
- **K value**: 5 neighbors
- **Process**:
  - Find 5 most similar samples
  - Impute missing value as weighted average
  - Preserves relationships between features
- **Applied to**: Outliers (set to NaN) and any naturally missing values

### 3. ✅ **Feature Scaling** (StandardScaler)
- **Method**: Z-score normalization
- **Formula**: z = (x - μ) / σ
- **Result**: Mean = 0, Std = 1 for all numeric features
- **Purpose**: Put all features on same scale for ML algorithms

### 4. ✅ **Categorical Encoding** (OneHotEncoder)
- **Method**: One-Hot Encoding
- **Categories Handled**:
  - Binary: yes/no → 2 columns
  - Binary: normal/abnormal → 2 columns
  - Binary: present/not present → 2 columns
  - Ternary: good/poor → 2 columns
  - Multi-class: low/moderate/high → 3 columns
- **Result**: 57 total features (28 numeric + 29 encoded categorical)
- **Handles Unknown**: Gracefully handles new categories at prediction time

### 5. ✅ **Class Balancing** (SMOTE)
- **Method**: Synthetic Minority Over-sampling Technique
- **Strategy**: Balance all classes to majority class size
- **Before**:
  - No_Disease: 13,145
  - Low_Risk: 1,643
  - Moderate_Risk: 657
  - High_Risk: 657
  - Severe_Disease: 328
- **After**:
  - All classes: 13,145 samples each
  - Total: 65,725 training samples
- **Applied to**: Training set only (test set preserved)

---

## 📈 Final Dataset Statistics

### Training Set (After Preprocessing)
- **Samples**: 65,725 (balanced via SMOTE)
- **Features**: 57
- **Class Distribution**: Perfectly balanced (13,145 each)

### Test Set (Original Distribution)
- **Samples**: 4,108 (20% of original)
- **Features**: 57
- **Class Distribution**: Preserved original imbalance
  - No_Disease: ~3,286 (80%)
  - Low_Risk: ~411 (10%)
  - Moderate_Risk: ~164 (4%)
  - High_Risk: ~164 (4%)
  - Severe_Disease: ~82 (2%)

---

## 💻 Usage

### Basic Usage

```python
from preprocessing.kidney_preprocessor import preprocess_kidney_data

# Preprocess data
X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
    data_path='data/raw/kidney_disease_dataset.csv',
    target='Target',
    test_size=0.2,
    handle_imbalance=True
)

print(f"Training shape: {X_train.shape}")  # (65725, 57)
print(f"Test shape: {X_test.shape}")        # (4108, 57)
```

### Advanced Configuration

```python
# Custom split and no balancing
X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
    data_path='data/raw/kidney_disease_dataset.csv',
    target='Target',
    test_size=0.3,              # 30% test set
    handle_imbalance=False,     # No SMOTE
    sampling_strategy='minority' # Only balance minority classes
)
```

### Using the Preprocessor Object

```python
from preprocessing.kidney_preprocessor import KidneyPreprocessor

# Create preprocessor
preprocessor = KidneyPreprocessor(
    numerical_features=['Age', 'Blood pressure (mm/Hg)', ...],
    categorical_features=['Hypertension (yes/no)', ...],
    target_column='Target',
    handle_imbalance=True
)

# Fit and transform
X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)
```

---

## 🎯 Key Features

### Automated Feature Detection
- Automatically identifies numeric vs categorical features
- Handles multiple categorical types
- Flexible column naming

### Robust Outlier Handling
- IQR method prevents influence of extreme values
- Outliers replaced with NaN then imputed
- Preserves data distribution

### No Data Leakage
- Preprocessing fitted on training data only
- Test data transformed using training statistics
- SMOTE applied only to training set

### Multi-Class Support
- Handles 5-class classification
- Balanced training for all classes
- Preserves test set distribution for realistic evaluation

---

## 📊 Verification

### Run Preprocessing Test

```bash
python3 preprocessing/kidney_preprocessor.py
```

**Expected Output:**
```
======================================================================
KIDNEY DISEASE DATA PREPROCESSING
======================================================================

📂 Loading data from: data/raw/kidney_disease_dataset.csv
✅ Loaded: 20538 samples, 43 features

📊 Feature types:
  Numeric: 28
  Categorical: 14

❓ Missing values:
  None detected

✂️  Splitting data (80/20)...
  Train: 16430 samples
  Test: 4108 samples

🔧 Preprocessing training data...

Class distribution before balancing:
  Class No_Disease: 13145
  Class Low_Risk: 1643
  Class Moderate_Risk: 657
  Class High_Risk: 657
  Class Severe_Disease: 328

Class distribution after SMOTE:
  Class No_Disease: 13145
  Class Low_Risk: 13145
  Class Moderate_Risk: 13145
  Class High_Risk: 13145
  Class Severe_Disease: 13145

🔧 Preprocessing test data...

✅ Preprocessing complete!
  Training shape: (65725, 57)
  Test shape: (4108, 57)
```

---

## 🏥 Clinical Features

### Demographics & Vitals
- Age, BMI, Blood Pressure

### Urine Tests
- Specific Gravity, Albumin, Sugar
- RBC, WBC, Pus cells, Bacteria
- Protein-to-creatinine ratio
- Urine output

### Blood Tests
- Glucose, Urea, Creatinine
- Sodium, Potassium, Calcium, Phosphate
- Hemoglobin, WBC count, RBC count
- Cholesterol, Albumin

### Advanced Markers
- eGFR (kidney function)
- Cystatin C
- Parathyroid Hormone (PTH)
- C-Reactive Protein (CRP)
- Interleukin-6 (IL-6)

### Medical History
- Hypertension, Diabetes, Coronary Artery Disease
- Family history of CKD
- Duration of diabetes/hypertension

### Lifestyle
- Smoking status
- Physical activity level
- Appetite, Pedal edema, Anemia

---

## ⚡ Performance Tips

### For Large Datasets
- KNN imputation can be slow on very large datasets
- Consider using SimpleImputer (mean/median) for faster processing
- Adjust `n_neighbors` based on data size

### For Imbalanced Data
- SMOTE works well for this dataset
- Can combine with undersampling for extreme imbalance
- Monitor for overfitting on minority classes

### For Feature Engineering
- Consider interaction terms (e.g., age × diabetes duration)
- Polynomial features for non-linear relationships
- Domain knowledge: eGFR is already a calculated feature

---

## 🚀 Next Steps

1. **Train Classification Models**
   ```bash
   python3 training/train_kidney_classifiers.py
   ```

2. **Evaluate Performance**
   - Accuracy, Precision, Recall, F1
   - Confusion Matrix
   - ROC-AUC for multi-class

3. **Feature Importance Analysis**
   - Which clinical markers matter most?
   - SHAP values for explainability

4. **Clinical Validation**
   - Verify model predictions align with medical knowledge
   - Test on external datasets

---

## ✅ Summary

**What's Done:**
- ✅ Comprehensive preprocessing pipeline
- ✅ 28 numeric features scaled
- ✅ 14 categorical features encoded (→ 57 total)
- ✅ Class balancing via SMOTE
- ✅ No data leakage
- ✅ Ready for model training

**Dataset Ready:**
- Training: 65,725 samples (balanced)
- Test: 4,108 samples (original distribution)
- Features: 57 (standardized and encoded)
- Classes: 5 (perfectly balanced in training)

**Production Ready:** Yes! 🎉

---

*Created: 2025-11-20*  
*Dataset: Chronic Kidney Disease (CKD)*  
*Samples: 20,538 → 69,833 (after SMOTE)*  
*Features: 42 → 57 (after encoding)*  
*Classes: 5 (balanced)*
