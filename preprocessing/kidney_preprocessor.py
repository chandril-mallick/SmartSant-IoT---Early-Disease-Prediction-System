"""
Kidney Disease Data Preprocessing Pipeline
Comprehensive preprocessing for Chronic Kidney Disease (CKD) classification.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using IQR method."""
    
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X = np.array(X)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - (self.threshold * iqr)
        self.upper_bounds_ = q3 + (self.threshold * iqr)
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("OutlierHandler has not been fitted yet")
        
        X = np.array(X)
        X_transformed = X.copy()
        
        for i in range(X.shape[1]):
            lower = self.lower_bounds_[i]
            upper = self.upper_bounds_[i]
            not_nan = ~np.isnan(X[:, i])
            outliers = not_nan & ((X[:, i] < lower) | (X[:, i] > upper))
            X_transformed[outliers, i] = np.nan
        
        return X_transformed


class KidneyPreprocessor:
    """Complete preprocessing pipeline for kidney disease data."""
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        target_column: str = 'Target',
        test_size: float = 0.2,
        random_state: int = 42,
        handle_imbalance: bool = True,
        sampling_strategy: str = 'auto'
    ):
        """
        Initialize the kidney disease preprocessor.
        
        Args:
            numerical_features: List of numeric feature names
            categorical_features: List of categorical feature names
            target_column: Name of target column
            test_size: Proportion for test set
            random_state: Random seed
            handle_imbalance: Whether to apply SMOTE
            sampling_strategy: SMOTE strategy ('auto', 'all', or dict)
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.handle_imbalance = handle_imbalance
        self.sampling_strategy = sampling_strategy
        
        # Initialize transformers
        self.outlier_handler = OutlierHandler(threshold=1.5)
        self.imputer = KNNImputer(n_neighbors=5, missing_values=np.nan)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # SMOTE for imbalance
        self.smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        self.undersampler = RandomUnderSampler(random_state=random_state)
        
        # Pipeline components
        self.preprocessor = None
    
    def _create_preprocessor(self):
        """Create the column transformer pipeline."""
        # Numeric pipeline: outlier -> impute -> scale
        numeric_transformer = Pipeline([
            ('outlier', self.outlier_handler),
            ('imputer', self.imputer),
            ('scaler', self.scaler)
        ])
        
        # Categorical pipeline: impute -> encode
        categorical_transformer = Pipeline([
            ('imputer', self.cat_imputer),
            ('encoder', self.encoder)
        ])
        
        # Combine
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform data with optional class balancing.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Transformed X and y arrays
        """
        if self.preprocessor is None:
            self._create_preprocessor()
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        y_transformed = y.values if hasattr(y, 'values') else np.array(y)
        
        # Handle class imbalance on training data
        if self.handle_imbalance:
            print(f"\nClass distribution before balancing:")
            unique, counts = np.unique(y_transformed, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  Class {cls}: {count}")
            
            try:
                X_transformed, y_transformed = self.smote.fit_resample(X_transformed, y_transformed)
                
                print(f"\nClass distribution after SMOTE:")
                unique, counts = np.unique(y_transformed, return_counts=True)
                for cls, count in zip(unique, counts):
                    print(f"  Class {cls}: {count}")
            except Exception as e:
                print(f"⚠️  SMOTE failed: {e}")
                print("Continuing without class balancing")
        
        return X_transformed, y_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet")
        
        return self.preprocessor.transform(X)
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets with stratification."""
        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )


def preprocess_kidney_data(
    data_path: str,
    target: str = 'Target',
    test_size: float = 0.2,
    random_state: int = 42,
    handle_imbalance: bool = True,
    sampling_strategy: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, KidneyPreprocessor]:
    """
    Complete preprocessing pipeline for kidney disease data.
    
    Args:
        data_path: Path to CSV file
        target: Target column name
        test_size: Test set proportion
        random_state: Random seed
        handle_imbalance: Apply SMOTE
        sampling_strategy: SMOTE strategy
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    print("\n" + "="*70)
    print("KIDNEY DISEASE DATA PREPROCESSING")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Identify feature types
    categorical_keywords = ['yes', 'no', 'normal', 'abnormal', 'present', 'good', 'poor', 'low', 'moderate', 'high']
    
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col == target:
            continue
        
        if df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"\n📊 Feature types:")
    print(f"  Numeric: {len(numerical_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\n❓ Missing values:")
    if missing.sum() == 0:
        print("  None detected")
    else:
        print(missing[missing > 0])
    
    # Separate features and target
    X = df[numerical_features + categorical_features]
    y = df[target]
    
    # Create preprocessor
    preprocessor = KidneyPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target,
        test_size=test_size,
        random_state=random_state,
        handle_imbalance=handle_imbalance,
        sampling_strategy=sampling_strategy
    )
    
    # Split data
    print(f"\n✂️  Splitting data ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Fit and transform
    print(f"\n🔧 Preprocessing training data...")
    X_train, y_train = preprocessor.fit_transform(X_train, y_train)
    
    print(f"\n🔧 Preprocessing test data...")
    X_test = preprocessor.transform(X_test)
    y_test = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_path = os.path.join(data_config.RAW_DATA_DIR, 'kidney_disease_dataset.csv')
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_kidney_data(
        data_path=data_path,
        target='Target',
        test_size=0.2,
        random_state=42,
        handle_imbalance=True
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING TEST COMPLETE")
    print("="*70)
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
