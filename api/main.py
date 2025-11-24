from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import io
import os
import json
from PIL import Image

from config import api_config, model_config, data_config
from models.urine_model import create_urine_model
from models.stool_model import create_stool_model
from preprocessing.base_preprocessor import UrineDataPreprocessor, StoolImagePreprocessor

# Initialize FastAPI app
app = FastAPI(
    title=api_config.TITLE,
    description=api_config.DESCRIPTION,
    version=api_config.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessors
urine_preprocessor = None
stool_preprocessor = StoolImagePreprocessor(image_size=model_config.IMAGE_SIZE)
utri_model = None
ckd_model = None
stool_model = None

# Pydantic models for request/response validation
class UrineFeatures(BaseModel):
    leukocyte_esterase: Optional[float] = None
    nitrite: Optional[float] = None
    protein: Optional[float] = None
    blood: Optional[float] = None
    glucose: Optional[float] = None
    ketones: Optional[float] = None
    wbc_count: Optional[float] = None
    rbc_count: Optional[float] = None
    bacteria_count: Optional[float] = None
    ph: Optional[float] = None
    specific_gravity: Optional[float] = None
    creatinine: Optional[float] = None
    turbidity: Optional[float] = None
    conductivity: Optional[float] = None
    age: float
    sex: str  # 'male' or 'female'

class PredictionResult(BaseModel):
    prediction: float
    confidence: float
    class_label: str

class StoolPredictionResult(BaseModel):
    predictions: List[float]
    predicted_class: int
    confidence: float
    class_label: str

# Helper functions
def load_models():
    """Load all models and preprocessors."""
    global urine_preprocessor, uti_model, ckd_model, stool_model
    
    # Load urine preprocessor
    preprocessor_path = os.path.join(data_config.MODELS_DIR, 'urine_preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        urine_preprocessor = UrineDataPreprocessor(data_config.RAW_DATA_DIR)
        urine_preprocessor.load_preprocessor(preprocessor_path)
    
    # Load UTI model
    uti_model_path = os.path.join(data_config.MODELS_DIR, data_config.UTI_MODEL)
    if os.path.exists(uti_model_path):
        uti_model = create_urine_model(
            input_dim=len(model_config.URINE_FEATURES) + 1,  # +1 for one-hot encoded sex
            target='uti'
        )
        uti_model.load_state_dict(torch.load(uti_model_path, map_location=torch.device('cpu')))
        uti_model.eval()
    
    # Load CKD model
    ckd_model_path = os.path.join(data_config.MODELS_DIR, data_config.CKD_MODEL)
    if os.path.exists(ckd_model_path):
        ckd_model = create_urine_model(
            input_dim=len(model_config.URINE_FEATURES) + 1,  # +1 for one-hot encoded sex
            target='ckd'
        )
        ckd_model.load_state_dict(torch.load(ckd_model_path, map_location=torch.device('cpu')))
        ckd_model.eval()
    
    # Load stool model
    stool_model_path = os.path.join(data_config.MODELS_DIR, data_config.STOOL_MODEL)
    if os.path.exists(stool_model_path):
        stool_model = create_stool_model(num_classes=7)  # 7 classes for Bristol stool scale
        stool_model.load_state_dict(torch.load(stool_model_path, map_location=torch.device('cpu')))
        stool_model.eval()

# Load models at startup
@app.on_event("startup")
async def startup_event():
    load_models()

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "SmartSant IoT API is running"}

# Predict UTI from urine test results
@app.post("/predict/uti", response_model=PredictionResult)
async def predict_uti(features: UrineFeatures):
    if uti_model is None or urine_preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame for preprocessing
        df = pd.DataFrame([features.dict()])
        
        # Preprocess features
        X, _ = urine_preprocessor.preprocess(df, target='uti')
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values)
        
        # Make prediction
        with torch.no_grad():
            logits = uti_model(X_tensor)
            prob = torch.sigmoid(logits).item()
        
        # Prepare response
        return {
            "prediction": prob,
            "confidence": max(prob, 1 - prob),
            "class_label": "UTI" if prob > 0.5 else "No UTI"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Predict CKD from urine test results
@app.post("/predict/ckd", response_model=PredictionResult)
async def predict_ckd(features: UrineFeatures):
    if ckd_model is None or urine_preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame for preprocessing
        df = pd.DataFrame([features.dict()])
        
        # Preprocess features
        X, _ = urine_preprocessor.preprocess(df, target='ckd')
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values)
        
        # Make prediction
        with torch.no_grad():
            logits = ckd_model(X_tensor)
            prob = torch.sigmoid(logits).item()
        
        # Prepare response
        return {
            "prediction": prob,
            "confidence": max(prob, 1 - prob),
            "class_label": "CKD/Proteinuria" if prob > 0.5 else "No CKD/Proteinuria"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Predict stool type from image
@app.post("/predict/stool", response_model=StoolPredictionResult)
async def predict_stool(file: UploadFile = File(...)):
    if stool_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = stool_preprocessor.load_image(image)
        
        # Add batch dimension and convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            logits = stool_model(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
        
        # Prepare response
        return {
            "predictions": probs.tolist(),
            "predicted_class": pred_class + 1,  # Convert to 1-7 scale
            "confidence": confidence,
            "class_label": f"Type {pred_class + 1}"  # Bristol stool type 1-7
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Combined prediction endpoint
@app.post("/predict/all")
async def predict_all(
    urine_features: UrineFeatures,
    stool_image: Optional[UploadFile] = File(None)
):
    results = {}
    
    # Get UTI prediction
    uti_result = await predict_uti(urine_features)
    results["uti"] = uti_result
    
    # Get CKD prediction
    ckd_result = await predict_ckd(urine_features)
    results["ckd"] = ckd_result
    
    # Get stool prediction if image is provided
    if stool_image:
        try:
            stool_result = await predict_stool(stool_image)
            results["stool"] = stool_result
        except Exception as e:
            results["stool"] = {"error": str(e)}
    
    return results

# Explain UTI prediction using SHAP
@app.post("/explain/uti")
async def explain_uti(features: UrineFeatures):
    if uti_model is None or urine_preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame for preprocessing
        df = pd.DataFrame([features.dict()])
        
        # Preprocess features
        X, _ = urine_preprocessor.preprocess(df, target='uti')
        
        # TODO: Add SHAP explanation
        # This is a placeholder - implement actual SHAP explanation
        explanation = {
            "feature_importances": {
                "leukocyte_esterase": 0.25,
                "nitrite": 0.20,
                "protein": 0.15,
                # ... other features
            },
            "base_value": 0.5,
            "output_value": 0.7,
            "message": "SHAP explanation will be implemented here"
        }
        
        return explanation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Explain stool prediction using Grad-CAM
@app.post("/explain/stool")
async def explain_stool(file: UploadFile = File(...)):
    if stool_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # TODO: Add Grad-CAM explanation
        # This is a placeholder - implement actual Grad-CAM
        explanation = {
            "heatmap": "path/to/heatmap.png",  # Path to generated heatmap
            "predicted_class": 3,
            "confidence": 0.85,
            "message": "Grad-CAM visualization will be implemented here"
        }
        
        return explanation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
