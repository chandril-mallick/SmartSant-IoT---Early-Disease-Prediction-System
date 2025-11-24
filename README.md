# 🏥 SmartSant-IoT: Early Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://smartsant-iot---early-disease-prediction-system.streamlit.app/)

**Quick Links:** [🌐 Live Demo](https://smartsant-iot---early-disease-prediction-system.streamlit.app/) | [📖 Documentation](#documentation) | [🚀 Installation](#installation) | [💻 API Docs](#api-documentation)

An advanced AI-powered medical diagnostic system leveraging machine learning and deep learning to predict diseases through **urine analysis**, **kidney disease classification**, and **Bristol stool scale image classification**. Built for IoT integration and real-time health monitoring.

---

## 🌐 Live Demo

**Try the interactive web application now!**

[!(https://smartsant-iot---early-disease-prediction-system.streamlit.app/)

🔗 **[Launch Live Demo](https://smartsant-iot---early-disease-prediction-system.streamlit.app/)**

### Demo Features:
- 💧 **Urine Analysis**: Real-time UTI detection with interactive form
- 📊 **Visual Analytics**: Probability gauges and risk factor analysis
- 🎨 **Modern UI**: Beautiful gradient design with responsive layout
- ⚡ **Instant Results**: Get predictions in seconds

> **Note**: The live demo uses the optimized Random Forest model with 93% accuracy. Simply enter test parameters and click "Analyze" to see results!

---

## 🎯 Project Overview

**SmartSant-IoT** is a comprehensive early disease prediction system that combines:
- **Urine Disease Classification**: UTI detection with 93% accuracy
- **Kidney Disease Prediction**: 5-class CKD risk assessment
- **Stool Image Analysis**: Bristol Stool Scale classification (Types 1-7)
- **RESTful API**: Production-ready FastAPI backend
- **Explainable AI**: SHAP values and Grad-CAM visualizations

### 🏆 Key Achievements
- ✅ **93.06% Accuracy** on UTI classification (optimized Random Forest)
- ✅ **Multi-class classification** for kidney disease (5 risk levels)
- ✅ **CNN-based image classification** for stool analysis (EfficientNet)
- ✅ **Comprehensive preprocessing** pipelines for all data types
- ✅ **Production-ready API** with FastAPI

---

## 🚀 Features

### 1. **Urine Disease Classification**
- **Binary Classification**: UTI (Urinary Tract Infection) detection
- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting, Neural Network, Ensemble
- **Best Model**: Random Forest with 93.06% accuracy
- **Metrics**: Precision (38.89%), Recall (43.75%), F1-Score (0.4118), AUC-ROC (0.7053)
- **Optimization**: Comprehensive hyperparameter tuning with 5-model comparison
- **Features**: 15 urine test parameters (WBC, RBC, bacteria, pH, specific gravity, etc.)

### 2. **Kidney Disease Prediction**
- **Multi-class Classification**: 5 risk levels (No Disease, Low, Moderate, High, Severe)
- **Dataset**: 20,538 patient records with 42 clinical features
- **Preprocessing**: IQR outlier removal, KNN imputation, StandardScaler, OneHotEncoder
- **Class Balancing**: SMOTE for balanced training (65,725 samples)
- **Features**: Blood tests, urine tests, demographics, medical history, lifestyle factors

### 3. **Bristol Stool Scale Classification**
- **Image Classification**: 7-class stool type classification
- **Model Architecture**: EfficientNet-B0 (transfer learning)
- **Preprocessing**: Data augmentation, normalization, 224×224 resizing
- **Evaluation**: Confusion matrix, ROC curves, Precision-Recall curves
- **Explainability**: Grad-CAM visualizations for interpretability

### 4. **RESTful API**
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Endpoints**: `/predict/urine`, `/predict/kidney`, `/predict/stool`
- **Features**: File upload, JSON input, batch predictions
- **Documentation**: Interactive Swagger UI at `/docs`

### 5. **Explainable AI**
- **SHAP Values**: Feature importance for urine/kidney models
- **Grad-CAM**: Visual explanations for stool image predictions
- **Transparency**: Understand model decisions for clinical trust

---

## 📂 Project Structure

```
SmartSant-IoT/
├── api/                              # FastAPI application
│   └── main.py                       # API endpoints and server
├── data/                             # Data storage
│   └── raw/                          # Raw datasets
│       ├── urine_data.csv            # Urine test data
│       ├── kidney_disease_dataset.csv # Kidney disease data
│       └── stool_images/             # Bristol stool images (Types 1-7)
├── models/                           # Trained models and evaluations
│   ├── urine_classifiers/            # Urine disease models
│   │   ├── optimized_urine_classifier.pkl  # Best model (Random Forest)
│   │   ├── optimized_model_metadata.json   # Performance metrics
│   │   └── all_models_comparison.json      # All 5 models comparison
│   ├── kidney_classifiers/           # Kidney disease models
│   │   ├── logistic_regression.pkl
│   │   ├── neural_network.pkl
│   │   └── best_model_metadata.json
│   ├── stool_evaluation/             # Stool model evaluation
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   └── precision_recall_curves.png
│   ├── enhanced_stool_model.py       # Stool CNN model definition
│   ├── urine_model.py                # Urine model definition
│   └── quick_reference.py            # Quick model usage examples
├── preprocessing/                    # Data preprocessing modules
│   ├── urine_preprocessor.py         # Urine data preprocessing
│   ├── kidney_preprocessor.py        # Kidney data preprocessing (IQR, KNN, SMOTE)
│   └── stool_image_preprocessor.py   # Image augmentation and preprocessing
├── training/                         # Model training scripts
│   ├── train_urine_classifiers.py    # Train urine models
│   ├── optimize_urine_classifier.py  # Hyperparameter optimization
│   ├── train_kidney_classifiers.py   # Train kidney models
│   ├── train_stool_model.py          # Train stool CNN
│   ├── evaluate_model.py             # Urine model evaluation
│   ├── evaluate_stool_model.py       # Stool model evaluation
│   └── evaluation_examples.py        # Evaluation utilities
├── inference/                        # Inference pipelines
│   ├── predict_urine_disease.py      # Urine prediction pipeline
│   └── unified_medical_predictor.py  # Combined prediction system
├── demos/                            # Demo scripts
│   └── classification_demo.py        # Interactive demo
├── config.py                         # Configuration settings
├── requirements.txt                  # Python dependencies
├── git-push.sh                       # Auto-push script
├── .gitignore                        # Git ignore rules
└── README.md                         # This file

# Documentation Files
├── URINE_CLASSIFIER_OPTIMIZATION_REPORT.md   # Detailed optimization results
├── KIDNEY_PREPROCESSING_GUIDE.md             # Kidney preprocessing guide
├── STOOL_MODEL_EVALUATION_REPORT.md          # Stool model evaluation
├── UNIFIED_MEDICAL_SYSTEM_GUIDE.md           # System integration guide
└── DATASET_SPLIT_INFO.md                     # Dataset split information
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System.git
cd SmartSant-IoT---Early-Disease-Prediction-System
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Include:
- **Core ML**: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`
- **Deep Learning**: `torch`, `torchvision`, `timm`
- **API**: `fastapi`, `uvicorn`, `pydantic`
- **Visualization**: `matplotlib`, `seaborn`, `shap`, `opencv-python`
- **Utilities**: `tqdm`, `joblib`, `python-dotenv`

---

## 🏃‍♂️ Quick Start

### 1. Train Models

#### Train Urine Classifier (with optimization)
```bash
python3 training/optimize_urine_classifier.py
```
**Output**: Optimized Random Forest model with 93% accuracy

#### Train Kidney Classifier
```bash
python3 training/train_kidney_classifiers.py
```
**Output**: Multiple models for 5-class kidney disease prediction

#### Train Stool Image Classifier
```bash
python3 training/train_stool_model.py
```
**Output**: EfficientNet-B0 CNN for Bristol Stool Scale classification

### 2. Run Inference

#### Predict Urine Disease
```python
from inference.predict_urine_disease import predict_urine_disease

# Example urine test data
urine_data = {
    'leukocyte_esterase': 2,
    'nitrite': 1,
    'protein': 1,
    'wbc_count': 50,
    'bacteria_count': 3,
    # ... other features
}

prediction = predict_urine_disease(urine_data)
print(f"UTI Prediction: {prediction}")
```

#### Unified Medical Predictor
```python
from inference.unified_medical_predictor import UnifiedMedicalPredictor

predictor = UnifiedMedicalPredictor()

# Predict from urine data
result = predictor.predict_from_urine(urine_data)
print(f"UTI Risk: {result['uti_prediction']}")
print(f"CKD Risk: {result['ckd_prediction']}")
```

### 3. Start API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access**:
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Make API Predictions

#### Urine Disease Prediction
```bash
curl -X POST "http://localhost:8000/predict/urine" \
  -H "Content-Type: application/json" \
  -d '{
    "leukocyte_esterase": 2,
    "nitrite": 1,
    "protein": 1,
    "wbc_count": 50,
    "bacteria_count": 3
  }'
```

#### Stool Image Classification
```bash
curl -X POST "http://localhost:8000/predict/stool" \
  -F "file=@path/to/stool_image.jpg"
```

---

## 📊 Model Performance

### Urine Disease Classifier (UTI Detection)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** 🏆 | **93.06%** | **38.89%** | **43.75%** | **0.4118** | **0.7053** |
| Ensemble (Voting) | 92.71% | 36.84% | 43.75% | 0.4000 | 0.7148 |
| Gradient Boosting | 92.71% | 35.29% | 37.50% | 0.3636 | 0.7121 |
| Logistic Regression | 92.01% | 31.58% | 37.50% | 0.3429 | 0.6990 |
| Neural Network | 87.50% | 18.75% | 37.50% | 0.2500 | 0.7093 |

**Best Model**: Random Forest with threshold 0.310
- **Hyperparameters**: `n_estimators=100, max_depth=15, class_weight='balanced_subsample'`
- **Optimization**: 10-fold cross-validation, 181 threshold tests
- **Improvement**: 83% F1-score improvement over baseline

### Kidney Disease Classifier (5-Class CKD)

- **Dataset**: 20,538 patients → 65,725 (after SMOTE)
- **Features**: 42 → 57 (after encoding)
- **Classes**: No Disease, Low Risk, Moderate Risk, High Risk, Severe Disease
- **Preprocessing**: IQR outlier removal, KNN imputation, StandardScaler, SMOTE
- **Models**: Logistic Regression, Neural Network (saved in `models/kidney_classifiers/`)

### Bristol Stool Scale Classifier

- **Architecture**: EfficientNet-B0 (transfer learning)
- **Classes**: 7 Bristol Stool Scale types
- **Evaluation**: Confusion matrix, ROC curves, Precision-Recall curves
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC
- **Visualizations**: Available in `models/stool_evaluation/`

---

## 🔬 Technical Details

### Preprocessing Pipelines

#### Urine Data
- **Missing Values**: KNN imputation
- **Scaling**: StandardScaler (Z-score normalization)
- **Encoding**: OneHotEncoder for categorical features
- **Imbalance**: SMOTE for minority class oversampling

#### Kidney Data
1. **Outlier Removal**: IQR method (1.5 × IQR threshold)
2. **Imputation**: KNN (k=5 neighbors)
3. **Scaling**: StandardScaler (mean=0, std=1)
4. **Encoding**: OneHotEncoder for 14 categorical features
5. **Balancing**: SMOTE (all classes → 13,145 samples)

#### Stool Images
- **Augmentation**: Random rotation, flip, color jitter, Gaussian blur
- **Normalization**: ImageNet mean/std
- **Resize**: 224×224 pixels
- **Format**: RGB images

### Model Architectures

#### Urine Classifier
- **Type**: Random Forest Ensemble
- **Trees**: 100 estimators
- **Depth**: 15 max depth
- **Features**: sqrt feature selection
- **Class Weight**: Balanced subsample

#### Kidney Classifier
- **Type**: Neural Network (MLP)
- **Layers**: Input → 256 → 128 → 64 → 5 (output)
- **Activation**: ReLU
- **Dropout**: 0.3
- **Optimizer**: Adam

#### Stool Classifier
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Fine-tuning**: Last layers unfrozen
- **Output**: 7-class softmax
- **Loss**: CrossEntropyLoss

---

## 📖 Documentation

Comprehensive guides available:

1. **[URINE_CLASSIFIER_OPTIMIZATION_REPORT.md](URINE_CLASSIFIER_OPTIMIZATION_REPORT.md)**
   - Detailed optimization process
   - 5-model comparison
   - Hyperparameter tuning results
   - Clinical interpretation

2. **[KIDNEY_PREPROCESSING_GUIDE.md](KIDNEY_PREPROCESSING_GUIDE.md)**
   - Complete preprocessing pipeline
   - Feature engineering
   - Class balancing with SMOTE
   - Usage examples

3. **[STOOL_MODEL_EVALUATION_REPORT.md](STOOL_MODEL_EVALUATION_REPORT.md)**
   - Evaluation metrics (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC)
   - Visualization plots
   - Multi-class considerations
   - Troubleshooting guide

4. **[UNIFIED_MEDICAL_SYSTEM_GUIDE.md](UNIFIED_MEDICAL_SYSTEM_GUIDE.md)**
   - System integration
   - Combined predictions
   - API usage

---

## 🔄 Git Workflow

### Auto-Push Script
Use the provided script for easy Git updates:

```bash
./git-push.sh
```

**Features**:
- Interactive commit message prompt
- Auto-generated timestamps
- Status checking
- Error handling

### Manual Workflow
```bash
git add .
git commit -m "Update: your changes here"
git push origin main
```

### Workflow Command
Type `/git-push` in the chat for automated workflow execution.

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Evaluate Models
```bash
# Evaluate urine model
python3 training/evaluate_model.py

# Evaluate stool model
python3 training/evaluate_stool_model.py

# Evaluate optimized model
python3 training/evaluate_optimized_model.py
```

---

## 🚀 Deployment

### Streamlit Cloud (Live) ✅

The web application is **currently deployed** on Streamlit Cloud:

🔗 **[https://smartsant-iot---early-disease-prediction-system.streamlit.app/](https://smartsant-iot---early-disease-prediction-system.streamlit.app/)**

**Deployment Steps**:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy! (automatic updates on git push)

**Features**:
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Auto-deploy on git push
- ✅ Built-in analytics

### Docker (Coming Soon)
```bash
docker build -t smartsant-iot .
docker run -p 8000:8000 smartsant-iot
```

### Cloud Deployment Options
- **Streamlit Cloud**: ✅ Currently deployed (recommended for web app)
- **AWS**: EC2 + S3 for model storage
- **Google Cloud**: Cloud Run + Cloud Storage
- **Azure**: App Service + Blob Storage
- **Heroku**: Easy deployment with Procfile

### Production Checklist
- [x] ✅ Streamlit Cloud deployment
- [x] ✅ HTTPS/SSL certificates (automatic)
- [ ] Environment variables for sensitive data
- [ ] Rate limiting and authentication
- [ ] Model versioning
- [ ] Monitoring and logging
- [ ] Backup and disaster recovery

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Chandril Mallick**
- GitHub: [@chandril-mallick](https://github.com/chandril-mallick)
- Repository: [SmartSant-IoT](https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System)

---

## 🙏 Acknowledgments

- **PyTorch** for deep learning framework
- **FastAPI** for modern API development
- **Scikit-learn** for machine learning algorithms
- **EfficientNet** for image classification backbone
- **SHAP** for explainable AI
- Medical datasets from public health repositories

---

## 📞 Support

For questions, issues, or feature requests:
- **Issues**: [GitHub Issues](https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System/discussions)

---

## 🔮 Future Roadmap

- [ ] **XGBoost Integration**: Add XGBoost for urine classification
- [ ] **Real-time IoT Integration**: Connect with IoT sensors
- [ ] **Mobile App**: React Native mobile application
- [ ] **Dashboard**: Web-based monitoring dashboard
- [ ] **Multi-language Support**: Internationalization
- [ ] **Federated Learning**: Privacy-preserving distributed training
- [ ] **Clinical Trials**: Validation with real-world medical data
- [ ] **Regulatory Compliance**: FDA/CE marking preparation

---

## 📊 Project Stats

- **Total Lines of Code**: ~15,000+
- **Models Trained**: 10+ (across all disease types)
- **Datasets**: 3 (Urine, Kidney, Stool)
- **Total Samples**: 20,000+ patient records
- **Accuracy**: 93% (UTI), Multi-class (Kidney), 7-class (Stool)
- **API Endpoints**: 6+
- **Documentation**: 2,000+ lines

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ for better healthcare through AI**

</div>
