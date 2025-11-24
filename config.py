from pydantic import BaseModel
from typing import List, Optional
import os

class DataConfig(BaseModel):
    # Paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    
    # Urine data configuration
    URINE_CSV: str = "urine_data.csv"
    STOOL_IMAGES_DIR: str = "stool_images"
    
    # Model files
    UTI_MODEL: str = "uti_model.pth"
    CKD_MODEL: str = "ckd_model.pth"
    STOOL_MODEL: str = "stool_model.pth"

class ModelConfig(BaseModel):
    # Common
    SEED: int = 42
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    
    # Urine model
    URINE_FEATURES: List[str] = [
        'leukocyte_esterase', 'nitrite', 'protein', 'blood', 'glucose',
        'ketones', 'wbc_count', 'rbc_count', 'bacteria_count', 'ph',
        'specific_gravity', 'creatinine', 'turbidity', 'conductivity', 'age'
    ]
    CATEGORICAL_FEATURES: List[str] = ['sex']
    TARGET_COLUMNS: List[str] = ['uti', 'ckd']
    
    # Image model
    IMAGE_SIZE: int = 224
    MEAN: List[float] = [0.485, 0.456, 0.406]  # ImageNet
    STD: List[float] = [0.229, 0.224, 0.225]   # ImageNet
    
    # Training
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 50
    PATIENCE: int = 10
    
    # Model architecture
    HIDDEN_LAYERS: List[int] = [256, 128, 64]
    DROPOUT_RATE: float = 0.3

class APIConfig(BaseModel):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    TITLE: str = "SmartSant IoT API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for SmartSant IoT - Disease Prediction System"

# Create config instances
data_config = DataConfig()
model_config = ModelConfig()
api_config = APIConfig()

# Create necessary directories
os.makedirs(data_config.RAW_DATA_DIR, exist_ok=True)
os.makedirs(data_config.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(data_config.MODELS_DIR, exist_ok=True)
