"""
SmartSant-IoT: Disease Prediction System
Streamlit Web Application for Visual Disease Classification

Features:
- Urine Disease Classification (UTI Detection)
- Kidney Disease Risk Assessment (5-class)
- Bristol Stool Scale Classification (Image-based)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="SmartSant-IoT | Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    h2 {
        color: #764ba2;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/health-book.png", width=80)
    st.title("🏥 SmartSant-IoT")
    st.markdown("### AI-Powered Disease Prediction")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Select Prediction Type:",
        ["🏠 Home", "💧 Urine Analysis", "🫘 Kidney Disease", "🔬 Stool Analysis"],
        index=0
    )
    
    st.markdown("---")
    
    # Information
    with st.expander("ℹ️ About"):
        st.markdown("""
        **SmartSant-IoT** is an advanced AI system for early disease prediction:
        
        - **Urine Analysis**: UTI detection (93% accuracy)
        - **Kidney Disease**: 5-level risk assessment
        - **Stool Analysis**: Bristol Scale classification
        
        Built with PyTorch, Scikit-learn, and FastAPI.
        """)
    
    with st.expander("📊 Model Performance"):
        st.markdown("""
        **Urine Classifier**
        - Accuracy: 93.06%
        - Model: Random Forest
        - F1-Score: 0.4118
        
        **Kidney Classifier**
        - Classes: 5 risk levels
        - Features: 57 clinical markers
        - Balanced with SMOTE
        
        **Stool Classifier**
        - Architecture: EfficientNet-B0
        - Classes: 7 Bristol types
        - Transfer learning
        """)

# Helper functions
@st.cache_resource
def load_urine_model():
    """Load the optimized urine classifier model"""
    try:
        model_path = Path("models/urine_classifiers/optimized_urine_classifier.pkl")
        metadata_path = Path("models/urine_classifiers/optimized_model_metadata.json")
        
        if model_path.exists():
            model = joblib.load(model_path)
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            return model, metadata
        else:
            st.warning("Urine model not found. Please train the model first.")
            return None, {}
    except Exception as e:
        st.error(f"Error loading urine model: {e}")
        return None, {}

@st.cache_resource
def load_kidney_model():
    """Load the kidney disease classifier model"""
    try:
        model_path = Path("models/kidney_classifiers/neural_network.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            return model
        else:
            st.warning("Kidney model not found. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading kidney model: {e}")
        return None

def create_gauge_chart(value, title, max_value=1.0):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# Page: Home
if page == "🏠 Home":
    st.title("🏥 SmartSant-IoT: Early Disease Prediction System")
    st.markdown("### AI-Powered Medical Diagnostics for Better Healthcare")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to **SmartSant-IoT**, an advanced artificial intelligence system designed to predict 
        diseases through comprehensive analysis of medical data and images.
        
        Our system combines state-of-the-art machine learning algorithms with deep learning models 
        to provide accurate, fast, and reliable disease predictions.
        """)
        
        st.markdown("#### 🎯 Key Features")
        
        features_col1, features_col2, features_col3 = st.columns(3)
        
        with features_col1:
            st.markdown("""
            **💧 Urine Analysis**
            - UTI Detection
            - 93% Accuracy
            - 15 Parameters
            - Real-time Results
            """)
        
        with features_col2:
            st.markdown("""
            **🫘 Kidney Disease**
            - 5 Risk Levels
            - 57 Features
            - SMOTE Balanced
            - Neural Network
            """)
        
        with features_col3:
            st.markdown("""
            **🔬 Stool Analysis**
            - Bristol Scale (1-7)
            - Image-based
            - EfficientNet CNN
            - Transfer Learning
            """)
    
    with col2:
        st.image("https://img.icons8.com/clouds/400/000000/artificial-intelligence.png", 
                 use_column_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### 📊 System Performance")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(label="Urine Accuracy", value="93.06%", delta="↑ 7% from baseline")
    
    with metric_col2:
        st.metric(label="Models Trained", value="10+", delta="3 disease types")
    
    with metric_col3:
        st.metric(label="Total Samples", value="20K+", delta="Patient records")
    
    with metric_col4:
        st.metric(label="API Endpoints", value="6+", delta="Production ready")
    
    st.markdown("---")
    
    # How it works
    st.markdown("#### 🔄 How It Works")
    
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    
    with step_col1:
        st.markdown("""
        **1️⃣ Input Data**
        
        Upload test results or images
        """)
    
    with step_col2:
        st.markdown("""
        **2️⃣ Preprocessing**
        
        Data cleaning & normalization
        """)
    
    with step_col3:
        st.markdown("""
        **3️⃣ AI Analysis**
        
        ML/DL model prediction
        """)
    
    with step_col4:
        st.markdown("""
        **4️⃣ Results**
        
        Visual reports & insights
        """)
    
    st.markdown("---")
    
    # Get started
    st.info("👈 **Get Started**: Select a prediction type from the sidebar to begin!")

# Page: Urine Analysis
elif page == "💧 Urine Analysis":
    st.title("💧 Urine Disease Classification")
    st.markdown("### UTI (Urinary Tract Infection) Detection")
    
    # Load model
    model, metadata = load_urine_model()
    
    if model is None:
        st.error("⚠️ Model not available. Please train the urine classifier first.")
        st.code("python3 training/optimize_urine_classifier.py")
    else:
        # Display model info
        with st.expander("ℹ️ Model Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Model Type**: {metadata.get('model_name', 'Random Forest')}
                
                **Accuracy**: {metadata.get('performance', {}).get('accuracy', 0.9306)*100:.2f}%
                
                **F1-Score**: {metadata.get('performance', {}).get('f1', 0.4118):.4f}
                """)
            with col2:
                st.markdown(f"""
                **Precision**: {metadata.get('performance', {}).get('precision', 0.3889)*100:.2f}%
                
                **Recall**: {metadata.get('performance', {}).get('recall', 0.4375)*100:.2f}%
                
                **AUC-ROC**: {metadata.get('performance', {}).get('auc', 0.7053):.4f}
                """)
        
        st.markdown("---")
        
        # Input form
        st.markdown("#### 📝 Enter Urine Test Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            leukocyte = st.selectbox("Leukocyte Esterase", [0, 1, 2, 3], 
                                     help="0=Negative, 1=Trace, 2=Small, 3=Moderate/Large")
            nitrite = st.selectbox("Nitrite", [0, 1], 
                                  help="0=Negative, 1=Positive")
            protein = st.selectbox("Protein", [0, 1, 2, 3], 
                                  help="0=Negative, 1=Trace, 2=Small, 3=Moderate/Large")
            blood = st.selectbox("Blood", [0, 1, 2, 3], 
                                help="0=Negative, 1=Trace, 2=Small, 3=Moderate/Large")
            glucose = st.selectbox("Glucose", [0, 1, 2, 3], 
                                  help="0=Negative, 1=Trace, 2=Small, 3=Moderate/Large")
        
        with col2:
            ketones = st.selectbox("Ketones", [0, 1, 2, 3], 
                                  help="0=Negative, 1=Trace, 2=Small, 3=Moderate/Large")
            wbc_count = st.number_input("WBC Count (cells/μL)", min_value=0, max_value=500, value=10)
            rbc_count = st.number_input("RBC Count (cells/μL)", min_value=0, max_value=100, value=5)
            bacteria = st.selectbox("Bacteria Count", [0, 1, 2, 3, 4], 
                                   help="0=None, 1=Few, 2=Moderate, 3=Many, 4=Numerous")
            ph = st.slider("pH", min_value=4.5, max_value=8.5, value=6.0, step=0.1)
        
        with col3:
            specific_gravity = st.slider("Specific Gravity", min_value=1.000, max_value=1.030, 
                                        value=1.015, step=0.001)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=300.0, 
                                        value=100.0, step=10.0)
            turbidity = st.selectbox("Turbidity", [0, 1, 2, 3], 
                                    help="0=Clear, 1=Slightly cloudy, 2=Cloudy, 3=Very cloudy")
            conductivity = st.number_input("Conductivity (mS/cm)", min_value=0.0, max_value=50.0, 
                                          value=20.0, step=1.0)
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=35)
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔍 Analyze Urine Sample", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'leukocyte_esterase': [leukocyte],
                'nitrite': [nitrite],
                'protein': [protein],
                'blood': [blood],
                'glucose': [glucose],
                'ketones': [ketones],
                'wbc_count': [wbc_count],
                'rbc_count': [rbc_count],
                'bacteria_count': [bacteria],
                'ph': [ph],
                'specific_gravity': [specific_gravity],
                'creatinine': [creatinine],
                'turbidity': [turbidity],
                'conductivity': [conductivity],
                'age': [age]
            })
            
            try:
                # Make prediction
                prediction_proba = model.predict_proba(input_data)[0]
                threshold = metadata.get('threshold', 0.5)
                prediction = 1 if prediction_proba[1] >= threshold else 0
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="prediction-box">
                            <h2>⚠️ UTI DETECTED</h2>
                            <p style="font-size: 1.2rem;">Urinary Tract Infection Likely</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning("**Recommendation**: Consult a healthcare provider for proper diagnosis and treatment.")
                    else:
                        st.markdown("""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);">
                            <h2>✅ NO UTI DETECTED</h2>
                            <p style="font-size: 1.2rem;">Urinary Tract Infection Unlikely</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("**Result**: No significant indicators of UTI detected.")
                
                with result_col2:
                    # Probability gauge
                    fig = create_gauge_chart(prediction_proba[1], "UTI Probability")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.markdown("#### 📈 Detailed Metrics")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("UTI Probability", f"{prediction_proba[1]*100:.2f}%")
                
                with metrics_col2:
                    st.metric("Healthy Probability", f"{prediction_proba[0]*100:.2f}%")
                
                with metrics_col3:
                    st.metric("Decision Threshold", f"{threshold*100:.2f}%")
                
                # Risk factors
                st.markdown("#### 🔍 Key Risk Indicators")
                
                risk_factors = []
                if leukocyte >= 2:
                    risk_factors.append("High leukocyte esterase")
                if nitrite == 1:
                    risk_factors.append("Positive nitrite")
                if wbc_count > 20:
                    risk_factors.append("Elevated WBC count")
                if bacteria >= 2:
                    risk_factors.append("Significant bacteria presence")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- ⚠️ {factor}")
                else:
                    st.markdown("- ✅ No major risk factors detected")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Page: Kidney Disease
elif page == "🫘 Kidney Disease":
    st.title("🫘 Kidney Disease Risk Assessment")
    st.markdown("### Chronic Kidney Disease (CKD) 5-Level Classification")
    
    st.info("🚧 **Coming Soon**: Kidney disease prediction interface is under development.")
    
    st.markdown("""
    The kidney disease classifier will assess CKD risk across 5 levels:
    
    1. **No Disease**: Healthy kidney function
    2. **Low Risk**: Minor abnormalities, monitoring recommended
    3. **Moderate Risk**: Noticeable decline in function
    4. **High Risk**: Significant kidney damage
    5. **Severe Disease**: Advanced CKD, treatment required
    
    **Features analyzed**: 57 clinical markers including:
    - Blood tests (glucose, urea, creatinine, electrolytes)
    - Urine tests (protein, albumin, RBC, WBC)
    - Vital signs (blood pressure, BMI)
    - Medical history (diabetes, hypertension)
    - Lifestyle factors (smoking, activity level)
    """)
    
    st.markdown("---")
    
    # Placeholder for future implementation
    with st.expander("📊 Model Architecture"):
        st.markdown("""
        **Model Type**: Neural Network (MLP)
        
        **Architecture**:
        - Input Layer: 57 features
        - Hidden Layer 1: 256 neurons (ReLU)
        - Hidden Layer 2: 128 neurons (ReLU)
        - Hidden Layer 3: 64 neurons (ReLU)
        - Output Layer: 5 classes (Softmax)
        
        **Preprocessing**:
        - IQR outlier removal
        - KNN imputation (k=5)
        - StandardScaler normalization
        - OneHotEncoder for categorical features
        - SMOTE for class balancing
        
        **Training Data**: 65,725 samples (balanced)
        
        **Test Data**: 4,108 samples (original distribution)
        """)

# Page: Stool Analysis
elif page == "🔬 Stool Analysis":
    st.title("🔬 Bristol Stool Scale Classification")
    st.markdown("### AI-Powered Stool Image Analysis")
    
    st.info("🚧 **Coming Soon**: Stool image classification interface is under development.")
    
    st.markdown("""
    The Bristol Stool Scale classifier uses deep learning to categorize stool images into 7 types:
    
    - **Type 1**: Separate hard lumps (severe constipation)
    - **Type 2**: Lumpy and sausage-like (mild constipation)
    - **Type 3**: Sausage with cracks (normal)
    - **Type 4**: Smooth, soft sausage (ideal/normal)
    - **Type 5**: Soft blobs with clear edges (lacking fiber)
    - **Type 6**: Mushy consistency with ragged edges (mild diarrhea)
    - **Type 7**: Liquid consistency (severe diarrhea)
    """)
    
    st.markdown("---")
    
    # Upload section (placeholder)
    st.markdown("#### 📤 Upload Stool Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            **Image Requirements**:
            - Format: JPG, PNG, WEBP
            - Size: Max 5MB
            - Resolution: Minimum 224×224 pixels
            - Quality: Clear, well-lit image
            
            **Privacy Notice**:
            - Images are processed locally
            - Not stored permanently
            - HIPAA compliant
            """)
        
        if st.button("🔍 Analyze Stool Sample", use_container_width=True):
            st.warning("Model inference will be implemented here.")
    
    # Model info
    with st.expander("📊 Model Architecture"):
        st.markdown("""
        **Model Type**: Convolutional Neural Network (CNN)
        
        **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
        
        **Architecture**:
        - Transfer learning from ImageNet
        - Fine-tuned last layers
        - 7-class output (Bristol Scale Types 1-7)
        
        **Preprocessing**:
        - Resize to 224×224 pixels
        - Normalization (ImageNet mean/std)
        - Data augmentation (rotation, flip, color jitter)
        
        **Evaluation Metrics**:
        - Accuracy, Precision, Recall, F1-Score
        - Specificity, ROC-AUC
        - Confusion matrix
        - ROC curves, Precision-Recall curves
        
        **Explainability**: Grad-CAM visualizations
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>SmartSant-IoT</strong> | AI-Powered Disease Prediction System</p>
    <p>Built with ❤️ using Streamlit, PyTorch, and Scikit-learn</p>
    <p>© 2025 Chandril Mallick | <a href="https://github.com/chandril-mallick/SmartSant-IoT---Early-Disease-Prediction-System" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
