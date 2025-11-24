"""
Urine Test Data Preprocessing Pipeline
Handles missing values, outliers, scaling, and class imbalance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import sys
import os

# Add project root to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Optional, Union, List

from config import data_config, model_config

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using IQR method."""
    
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        self.iqr = None
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X = np.array(X)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        self.iqr = q3 - q1
        self.lower_bounds_ = q1 - (self.threshold * self.iqr)
        self.upper_bounds_ = q3 + (self.threshold * self.iqr)
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("OutlierHandler has not been fitted yet")
            
        X = np.array(X)
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            lower = self.lower_bounds_[i]
            upper = self.upper_bounds_[i]
            # Only check non-NaN values
            # mask = (X[:, i] >= lower) & (X[:, i] <= upper)
            
            # Easier: set outliers to NaN
            not_nan = ~np.isnan(X[:, i])
            outliers = not_nan & ((X[:, i] < lower) | (X[:, i] > upper))
            X_transformed[outliers, i] = np.nan
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)


class UrinePreprocessor:
    """Complete preprocessing pipeline for urine test data."""
    
    def __init__(
        self,
        numerical_features: List[str] = [
            'Age', 'pH', 'Specific Gravity', 'WBC', 'RBC',
            'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates', 'Bacteria',
            'leukocyte_esterase', 'nitrite', 'protein', 'blood', 'glucose',
            'ketones', 'wbc_count', 'rbc_count', 'bacteria_count', 'ph',
            'specific_gravity', 'creatinine', 'turbidity', 'conductivity'
        ],
        categorical_features: List[str] = [
            'Gender', 'Color', 'Transparency', 'Glucose', 'Protein',
            'sex', 'color', 'transparency', 'glucose', 'protein'
        ],
        target_columns: List[str] = ['Diagnosis', 'uti'],
        test_size: float = 0.2,
        random_state: int = 42,
        handle_imbalance: bool = True,
        sampling_strategy: str = 'auto',  # 'over', 'under', or 'auto'
        save_path: Optional[str] = None
    ):
        # Clean and deduplicate features
        self.numerical_features = list(set([f.strip() for f in numerical_features if f]))
        self.categorical_features = list(set([f.strip() for f in categorical_features if f]))
        self.target_columns = list(set([t.strip() for t in target_columns if t]))
        
        # Remove any overlap between numerical and categorical features
        self.numerical_features = [f for f in self.numerical_features if f not in self.categorical_features]
        
        self.test_size = test_size
        self.random_state = random_state
        self.handle_imbalance = handle_imbalance
        self.sampling_strategy = sampling_strategy
        self.save_path = save_path or os.path.join(data_config.MODELS_DIR, 'urine_preprocessor.pkl')
        
        # Initialize transformers
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5, missing_values=np.nan)
        self.outlier_handler = OutlierHandler(threshold=1.5)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        self.undersampler = RandomUnderSampler(random_state=random_state)
        
        # Track feature names for debugging
        self.feature_names_out_ = None
        
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create the preprocessing pipeline."""
        # Numeric transformations
        numeric_transformer = Pipeline([
            ('outlier', self.outlier_handler),  # Handle outliers first (introduces NaNs)
            ('imputer', self.imputer),          # Impute missing values (including outliers)
            ('scaler', self.scaler)             # Scale data
        ])
        
        # Categorical transformations
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ('encoder', self.encoder)
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numerical_features) if self.numerical_features else ('num', 'passthrough', []),
            ('cat', categorical_transformer, self.categorical_features) if self.categorical_features else ('cat', 'passthrough', [])
        ], remainder='drop')
        
        return preprocessor
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'UrinePreprocessor':
        """Fit the preprocessing pipeline."""
        # Make a copy to avoid modifying the original data
        X = X.copy()
        
        # Print info about missing values
        missing_values = X.isna().sum()
        print("\nMissing values per column before preprocessing:")
        print(missing_values[missing_values > 0])
        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        
        try:
            # Fit and transform features
            X_processed = self.preprocessor.fit_transform(X, y)
            
            # Handle class imbalance if specified and y is provided
            if y is not None and self.handle_imbalance:
                # Check for NaN values in y
                if y.isna().any():
                    print("Warning: Found NaN values in target variable. Dropping these rows.")
                    valid_idx = y.notna()
                    X_processed = X_processed[valid_idx]
                    y = y[valid_idx]
                
                # Get class distribution
                class_counts = y.value_counts()
                print("\nClass distribution before balancing:")
                print(class_counts)
                
                # Determine sampling strategy
                if len(class_counts) > 1:  # Only balance if we have multiple classes
                    if self.sampling_strategy == 'auto':
                        if min(class_counts) < 100:  # If minority class is very small, use SMOTE
                            X_processed, y = self.smote.fit_resample(X_processed, y)
                        else:  # Otherwise, use random undersampling
                            X_processed, y = self.undersampler.fit_resample(X_processed, y)
                    elif self.sampling_strategy == 'over':
                        X_processed, y = self.smote.fit_resample(X_processed, y)
                    elif self.sampling_strategy == 'under':
                        X_processed, y = self.undersampler.fit_resample(X_processed, y)
                    
                    print("\nClass distribution after balancing:")
                    print(pd.Series(y).value_counts())
            
            # Save the preprocessor
            self.save()
            
            return X_processed, y
            
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape if y is not None else 'None'}")
            print(f"X columns: {X.columns.tolist()}")
            print(f"y type: {type(y) if y is not None else 'None'}")
            raise
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet")
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform the data."""
        return self.fit(X, y)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the preprocessor to disk."""
        save_path = path or self.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self, save_path)
        
    @classmethod
    def load(cls, path: str) -> 'UrinePreprocessor':
        """Load a pre-trained preprocessor from disk."""
        return joblib.load(path)
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets with stratification."""
        test_size = test_size or self.test_size
        random_state = random_state or self.random_state
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )


def convert_range_to_mean(value):
    """Convert range string or qualitative description to numeric value."""
    if pd.isna(value):
        return np.nan
        
    value_str = str(value).strip().upper()
    
    # Qualitative mappings
    mappings = {
        'NONE SEEN': 0.0,
        'NONE': 0.0,
        'RARE': 1.5,      # Assumed small number
        'OCCASIONAL': 3.0, # Assumed slightly larger
        'FEW': 5.0,
        'MODERATE': 10.0,
        'PLENTY': 20.0,
        'LOADED': 30.0,
        'TNTC': 50.0,     # Too Numerous To Count
        'TRACE': 0.5
    }
    
    if value_str in mappings:
        return mappings[value_str]
        
    # Handle inequalities
    if '>' in value_str:
        try:
            return float(value_str.replace('>', '')) + 1.0
        except ValueError:
            pass
    if '<' in value_str:
        try:
            return max(0.0, float(value_str.replace('<', '')) - 1.0)
        except ValueError:
            pass
            
    # Handle ranges (e.g., "1-3")
    if '-' in value_str:
        try:
            parts = value_str.split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, AttributeError):
            pass
            
    # Handle direct numbers
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def preprocess_urine_data(
    data_path: str,
    target: str = 'Diagnosis',
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    imputer_strategy: str = 'knn',
    outlier_threshold: float = 1.5,
    handle_imbalance: bool = True,
    sampling_strategy: str = 'auto',
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 'UrinePreprocessor']:
    """
    Complete preprocessing pipeline for urine test data.
    
    Args:
        data_path: Path to the raw urine test data CSV file
        target: Target variable to predict
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        imputer_strategy: Strategy for imputing missing values
        outlier_threshold: Threshold for detecting outliers
        handle_imbalance: Whether to handle class imbalance
        sampling_strategy: Strategy for handling imbalance ('over', 'under', or 'auto')
        save_path: Path to save the preprocessor
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load data first to check available columns
    df = pd.read_csv(data_path)
    
    # Print available columns for debugging
    print("Available columns in the dataset:", df.columns.tolist())
    
    # Define default numerical and categorical features based on actual data
    available_columns = list(df.columns)  # Convert to list to ensure indexing works
    
    # Try to match columns case-insensitively
    def find_matching_columns(possible_columns, available_columns):
        available_lower = [col.lower() for col in available_columns]
        matched = []
        for col in possible_columns:
            col_lower = col.lower()
            if col in available_columns:
                matched.append(col)
            elif col_lower in available_lower:
                # Find the correct case of the column
                idx = available_lower.index(col_lower)
                matched.append(available_columns[idx])
        return matched
    
    # Define possible column names
    possible_numerical = [
        'Age', 'pH', 'Specific Gravity', 'WBC', 'RBC',
        'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates', 'Bacteria',
        'leukocyte_esterase', 'nitrite', 'protein', 'blood', 'glucose',
        'ketones', 'wbc_count', 'rbc_count', 'bacteria_count', 'ph',
        'specific_gravity', 'creatinine', 'turbidity', 'conductivity', 'age'
    ]
    
    possible_categorical = [
        'Gender', 'Color', 'Transparency', 'Glucose', 'Protein',
        'sex', 'color', 'transparency', 'glucose', 'protein'
    ]
    
    # Find matching columns
    matched_numerical = find_matching_columns(possible_numerical, available_columns)
    matched_categorical = find_matching_columns(possible_categorical, available_columns)
    
    # Use provided features or matched defaults
    numerical_features = numerical_features or matched_numerical
    categorical_features = categorical_features or matched_categorical
    
    # Ensure no overlap between numerical and categorical features
    numerical_features = [f for f in numerical_features if f not in categorical_features]
    
    # Ensure target is not in features
    if target in numerical_features:
        numerical_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)
    
    print(f"Using numerical features: {numerical_features}")
    print(f"Using categorical features: {categorical_features}")
    
    # Check if we have any features to work with
    if not numerical_features and not categorical_features:
        raise ValueError("No valid features found in the dataset. Please check column names.")
    
    # Ensure all specified columns exist in the dataframe
    missing_cols = set(numerical_features + categorical_features + [target]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"The following columns were not found in the dataset: {missing_cols}")
    
    # Convert target to string if it's categorical
    if df[target].dtype == 'object':
        df[target] = df[target].astype(str)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Convert range values to mean for numerical columns
    for col in ['WBC', 'RBC', 'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates', 'Bacteria']:
        if col in df.columns:
            df[col] = df[col].apply(convert_range_to_mean)
    
    # Convert specific gravity to float (handle '1.0' prefix if present)
    if 'Specific Gravity' in df.columns:
        df['Specific Gravity'] = df['Specific Gravity'].astype(str).str.replace('1.0', '').astype(float)
    
    # Convert Age to numeric (handle any non-numeric values)
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Select features and target
    # Use set to remove duplicates while preserving order (in Python 3.7+)
    all_features = numerical_features + categorical_features
    unique_features = []
    seen = set()
    for f in all_features:
        if f in df.columns and f not in seen:
            unique_features.append(f)
            seen.add(f)
            
    X = df[unique_features]
    y = df[target].apply(lambda x: 1 if str(x).strip().upper() in ['POSITIVE', 'YES', 'Y', '1', 'TRUE'] else 0)
    
    # Initialize preprocessor
    preprocessor = UrinePreprocessor(
        numerical_features=[f for f in numerical_features if f in df.columns],
        categorical_features=[f for f in categorical_features if f in df.columns],
        target_columns=[target],
        test_size=test_size,
        random_state=random_state,
        handle_imbalance=handle_imbalance,
        sampling_strategy=sampling_strategy,
        save_path=save_path
    )
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Fit and transform training data
    X_train, y_train = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join('data', 'raw', 'urine_data.csv')
    
    try:
        # Preprocess data
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessor = preprocess_urine_data(
            data_path=data_path,
            target='Diagnosis',
            save_path=os.path.join(data_config.MODELS_DIR, 'urine_preprocessor.pkl')
        )
        
        print("\nPreprocessing complete!")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Class distribution - Test: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
