from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

from config import data_config, model_config

class BasePreprocessor(ABC):
    """Base class for data preprocessing."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.preprocessor = None
        self.feature_columns = None
        
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw data from the specified path."""
        pass
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the data.
        
        Args:
            df: Raw input data
            
        Returns:
            Tuple of (features, target) where features is a DataFrame and target is a Series
        """
        pass
    
    def save_preprocessor(self, path: str):
        """Save the fitted preprocessor to disk."""
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, path)
    
    def load_preprocessor(self, path: str):
        """Load a pre-trained preprocessor from disk."""
        if os.path.exists(path):
            self.preprocessor = joblib.load(path)
        return self.preprocessor


class UrineDataPreprocessor(BasePreprocessor):
    """Preprocessor for urine test data."""
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.numerical_features = [
            'leukocyte_esterase', 'nitrite', 'protein', 'blood', 'glucose',
            'ketones', 'wbc_count', 'rbc_count', 'bacteria_count', 'ph',
            'specific_gravity', 'creatinine', 'turbidity', 'conductivity', 'age'
        ]
        self.categorical_features = ['sex']
        self.target_columns = ['uti', 'ckd']
    
    def load_data(self) -> pd.DataFrame:
        """Load urine test data from CSV."""
        return pd.read_csv(os.path.join(self.data_path, data_config.URINE_CSV))
    
    def preprocess(self, df: pd.DataFrame, target: str = 'uti') -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess urine test data.
        
        Args:
            df: Raw urine test data
            target: Target variable ('uti' or 'ckd')
            
        Returns:
            Tuple of (features, target) where features is a DataFrame and target is a Series
        """
        if target not in self.target_columns:
            raise ValueError(f"Target must be one of {self.target_columns}")
            
        # Separate features and target
        X = df[self.numerical_features + self.categorical_features].copy()
        y = df[target].copy()
        
        # Create preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, add_indicator=False)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        num_features = self.numerical_features
        cat_features = []
        if hasattr(self.preprocessor.named_transformers_['cat'].named_steps['encoder'], 'get_feature_names_out'):
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features)
        
        self.feature_columns = list(num_features) + list(cat_features)
        
        return pd.DataFrame(X_processed, columns=self.feature_columns), y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )


class StoolImagePreprocessor:
    """Preprocessor for stool images."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.classes = [1, 2, 3, 4, 5, 6, 7]  # Bristol stool scale
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        import cv2
        from PIL import Image
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Convert to PyTorch format (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to an image."""
        import cv2
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            
        # Random rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Random brightness and contrast
        alpha = random.uniform(0.9, 1.1)  # Contrast control
        beta = random.uniform(-0.1, 0.1)  # Brightness control
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image
