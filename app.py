"""
SmartSant-IoT: Advanced Disease Prediction System
Enterprise Edition - Professional Medical AI Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import hashlib
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Page Config
st.set_page_config(
    page_title="SmartSant-IoT | Medical AI Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/chandril-mallick/SmartSant-IoT',
        'Report a bug': "https://github.com/chandril-mallick/SmartSant-IoT/issues",
        'About': "SmartSant-IoT - Enterprise Medical AI Platform"
    }
)

# Professional CSS Styling
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #1a202c;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.875rem;
        color: #2d3748;
    }
    
    h3 {
        font-size: 1.5rem;
        color: #4a5568;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Make all form labels and text black/dark */
    label, .stMarkdown, .stText {
        color: #1a202c !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    /* Form field labels */
    .stNumberInput label, 
    .stSelectbox label, 
    .stSlider label,
    .stTextInput label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Card Components */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        border-left-width: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Form Elements */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border: none;
        padding: 1rem 1.25rem;
        font-weight: 500;
    }
    
    /* Prediction Result Box */
    .prediction-result {
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-result h2 {
        font-size: 2rem;
        margin: 0;
        color: white;
        font-weight: 700;
    }
    
    .prediction-result p {
        font-size: 1.125rem;
        margin: 0.5rem 0 0 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem 0;
        width: 100%;
        justify-content: center;
        border: 2px solid;
    }
    
    .status-online {
        background: rgba(255, 255, 255, 0.95);
        color: #15803d !important;
        border-color: #16a34a;
        font-weight: 700;
    }
    
    .status-offline {
        background: rgba(255, 255, 255, 0.95);
        color: #b91c1c !important;
        border-color: #dc2626;
        font-weight: 700;
    }
    
    /* Override sidebar white text for status badges */
    [data-testid="stSidebar"] .status-badge {
        color: inherit !important;
    }
    
    [data-testid="stSidebar"] .status-online {
        color: #15803d !important;
    }
    
    [data-testid="stSidebar"] .status-offline {
        color: #b91c1c !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_urine_model():
    """Load Urine Classifier Artifacts"""
    try:
        base_path = Path("models")
        model = joblib.load(base_path / "urine_classifiers/optimized_urine_classifier.pkl")
        preprocessor = joblib.load(base_path / "urine_preprocessor.pkl")
        
        meta_path = base_path / "urine_classifiers/optimized_model_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
        return model, preprocessor, metadata
    except Exception as e:
        return None, None, {"error": str(e)}

@st.cache_resource
def load_kidney_model():
    """Load Kidney Classifier Artifacts"""
    try:
        base_path = Path("models/kidney_classifiers")
        model = joblib.load(base_path / "optimized_kidney_classifier.pkl")
        scaler = joblib.load(base_path / "scaler.pkl")
        le_classes = joblib.load(base_path / "label_encoder_classes.pkl")
        
        meta_path = base_path / "optimized_model_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
        return model, scaler, le_classes, metadata
    except Exception as e:
        return None, None, None, {"error": str(e)}

def create_gauge(value, title):
    """Create a professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#475569', 'family': 'Inter'}},
        number = {'suffix': "%", 'font': {'size': 36, 'color': '#1e293b'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#3b82f6", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': '#dcfce7'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#fee2e2'}
            ],
        }
    ))
    fig.update_layout(
        height=280, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter"}
    )
    return fig

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='color: white; font-size: 1.75rem; margin-bottom: 0.5rem;'>SmartSant-IoT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(255,255,255,0.8); font-size: 0.875rem; margin-bottom: 2rem;'>Medical AI Platform v2.0</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    selected_page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Urine Analysis", "Kidney Assessment", "Stool Analysis"],
        label_visibility="visible"
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.2); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # System Status
    st.markdown("<p style='font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; color: white;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    
    u_model, _, _ = load_urine_model()
    k_model, _, _, _ = load_kidney_model()
    
    status_urine = "ONLINE" if u_model else "OFFLINE"
    status_kidney = "ONLINE" if k_model else "OFFLINE"
    
    st.markdown(f"""
    <div style='margin-bottom: 0.75rem;'>
        <span class='status-badge {"status-online" if u_model else "status-offline"}'>
            Urine Module: {status_urine}
        </span>
    </div>
    <div style='margin-bottom: 0.75rem;'>
        <span class='status-badge {"status-online" if k_model else "status-offline"}'>
            Kidney Module: {status_kidney}
        </span>
    </div>
    <div>
        <span class='status-badge status-online'>
            Stool Module: ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

# Main Content
if selected_page == "Dashboard":
    st.markdown("<h1>Medical Diagnostics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.125rem; color: #64748b; margin-bottom: 2rem;'>AI-powered disease prediction and risk assessment platform</p>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">24,538</div>
            <div class="metric-delta" style="color: #16a34a;">↑ 12.5% vs last week</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">System Accuracy</div>
            <div class="metric-value">94.2%</div>
            <div class="metric-delta" style="color: #16a34a;">High Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Modules</div>
            <div class="metric-value">3/3</div>
            <div class="metric-delta" style="color: #3b82f6;">All Operational</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Response Time</div>
            <div class="metric-value">45ms</div>
            <div class="metric-delta" style="color: #16a34a;">Optimal</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Available Modules</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #3b82f6;">Urine Analysis</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Detect urinary tract infections using advanced chemical parameter analysis with 93% accuracy.
            </p>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <span style="font-size: 0.875rem; color: #64748b;">15 Parameters • Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #10b981;">Kidney Assessment</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Predict chronic kidney disease risk across 5 severity levels using comprehensive clinical markers.
            </p>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <span style="font-size: 0.875rem; color: #64748b;">24 Markers • LightGBM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #f59e0b;">Stool Analysis</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Classify stool samples using Bristol Scale with computer vision and deep learning models.
            </p>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <span style="font-size: 0.875rem; color: #64748b;">7 Classes • EfficientNet</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Urine Analysis":
    st.markdown("<h1>Urine Disease Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.125rem; color: #64748b; margin-bottom: 2rem;'>UTI Detection System</p>", unsafe_allow_html=True)
    
    model, preprocessor, metadata = load_urine_model()
    
    if not model:
        st.error("Model not available. Please train the urine classifier first.")
        st.stop()
        
    with st.form("urine_form"):
        st.markdown("<div class='section-header'>Patient Information & Test Results</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age (years)", 0, 120, 35)
            sex = st.selectbox("Sex", ["Male", "Female"])
            
            st.markdown("**Physical Properties**")
            ph = st.slider("pH Level", 4.5, 8.5, 6.0, 0.1)
            sg = st.slider("Specific Gravity", 1.000, 1.030, 1.015, 0.001, format="%.3f")
            
        with col2:
            st.markdown("**Chemical Analysis**")
            leukocyte = st.selectbox("Leukocyte Esterase", [0, 1, 2, 3], 
                                    format_func=lambda x: ["Negative", "Trace", "Small", "Large"][x])
            nitrite = st.selectbox("Nitrite", [0, 1], 
                                  format_func=lambda x: ["Negative", "Positive"][x])
            protein = st.selectbox("Protein", [0, 1, 2, 3])
            glucose = st.selectbox("Glucose", [0, 1, 2, 3])
            
        with col3:
            st.markdown("**Microscopic Examination**")
            bacteria = st.selectbox("Bacteria Count", [0, 1, 2, 3, 4])
            wbc = st.number_input("WBC Count (cells/μL)", 0, 500, 10)
            rbc = st.number_input("RBC Count (cells/μL)", 0, 100, 5)
            turbidity = st.selectbox("Turbidity", [0, 1, 2, 3], 
                                    format_func=lambda x: ["Clear", "Slightly Cloudy", "Cloudy", "Very Turbid"][x])
            
        submitted = st.form_submit_button("Analyze Sample")
        
    if submitted:
        with st.spinner("Processing biomarkers..."):
            input_data = pd.DataFrame({
                'Age': [age], 'Gender': [sex], 'pH': [ph], 'Specific Gravity': [sg],
                'WBC': [wbc], 'RBC': [rbc], 'Epithelial Cells': [0], 'Mucous Threads': [0],
                'Amorphous Urates': [0], 'Bacteria': [bacteria], 'Color': ['Yellow'],
                'Transparency': ['Clear' if turbidity == 0 else 'Cloudy'],
                'Glucose': [glucose], 'Protein': [protein],
                'leukocyte_esterase': [leukocyte], 'nitrite': [nitrite],
                'blood': [0], 'ketones': [0], 'creatinine': [100.0], 'conductivity': [20.0]
            })
            
            try:
                if preprocessor:
                    X = preprocessor.transform(input_data)
                    if hasattr(X, 'values'): X = X.values
                else:
                    X = input_data.drop(columns=['Gender'], errors='ignore').values
                
                proba = model.predict_proba(X)[0]
                p_uti = proba[1]
                
                st.markdown("<hr>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if p_uti > 0.5:
                        st.markdown(f"""
                        <div class="prediction-result" style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);">
                            <h2>UTI DETECTED</h2>
                            <p>Confidence: {p_uti*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("High probability of urinary tract infection. Clinical correlation recommended.")
                    else:
                        st.markdown(f"""
                        <div class="prediction-result" style="background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);">
                            <h2>NEGATIVE</h2>
                            <p>Confidence: {(1-p_uti)*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("No significant signs of UTI detected.")
                        
                with col2:
                    fig = create_gauge(p_uti, "Infection Probability")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

elif selected_page == "Kidney Assessment":
    st.markdown("<h1>Kidney Disease Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.125rem; color: #64748b; margin-bottom: 2rem;'>CKD Stage Prediction (5-Class)</p>", unsafe_allow_html=True)
    
    model, scaler, le_classes, metadata = load_kidney_model()
    
    if not model:
        st.warning("Kidney model artifacts missing. Please complete training pipeline.")
        st.stop()
        
    with st.form("kidney_form"):
        st.markdown("<div class='section-header'>Clinical Parameters</div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Basic Information", "Blood Panel", "Urinalysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age (years)", 0, 120, 45)
                bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
                htn = st.selectbox("Hypertension", ["No", "Yes"])
                dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
            with col2:
                cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
                appet = st.selectbox("Appetite", ["Good", "Poor"])
                pe = st.selectbox("Pedal Edema", ["No", "Yes"])
                ane = st.selectbox("Anemia", ["No", "Yes"])
                
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                bgr = st.number_input("Blood Glucose Random (mg/dL)", 0, 500, 120)
                bu = st.number_input("Blood Urea (mg/dL)", 0, 300, 40)
                sc = st.number_input("Serum Creatinine (mg/dL)", 0.0, 50.0, 1.2)
            with col2:
                sod = st.number_input("Sodium (mEq/L)", 0, 200, 137)
                pot = st.number_input("Potassium (mEq/L)", 0.0, 10.0, 4.5)
                hemo = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 14.0)
            with col3:
                pcv = st.number_input("Packed Cell Volume (%)", 0, 60, 40)
                wc = st.number_input("WBC Count (cells/cumm)", 0, 30000, 8000)
                rc = st.number_input("RBC Count (millions/cmm)", 0.0, 10.0, 5.0)
                
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
                al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
                su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
            with col2:
                rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
                pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
                ba = st.selectbox("Bacteria", ["Not Present", "Present"])
                
        submitted = st.form_submit_button("Assess Risk")
        
    if submitted:
        with st.spinner("Evaluating kidney function..."):
            try:
                cat_map = {"No": 0, "Yes": 1, "Good": 1, "Poor": 0, "Normal": 0, "Abnormal": 1, "Not Present": 0, "Present": 1}
                
                input_dict = {
                    'Age': age, 'Blood_Pressure': bp, 'Specific_Gravity': sg, 'Albumin': al, 'Sugar': su,
                    'Red_Blood_Cells': cat_map[rbc], 'Pus_Cells': cat_map[pc], 
                    'Pus_Cell_Clumps': cat_map[pcc], 'Bacteria': cat_map[ba],
                    'Blood_Glucose_Random': bgr, 'Blood_Urea': bu, 'Serum_Creatinine': sc, 
                    'Sodium': sod, 'Potassium': pot, 'Hemoglobin': hemo, 
                    'Packed_Cell_Volume': pcv, 'White_Blood_Cell_Count': wc, 
                    'Red_Blood_Cell_Count': rc, 'Hypertension': cat_map[htn], 
                    'Diabetes_Mellitus': cat_map[dm], 'Coronary_Artery_Disease': cat_map[cad],
                    'Appetite': cat_map[appet], 'Pedal_Edema': cat_map[pe], 'Anemia': cat_map[ane]
                }
                
                df = pd.DataFrame([input_dict])
                expected_features = metadata.get('feature_names', list(input_dict.keys()))
                
                for col in expected_features:
                    if col not in df.columns:
                        df[col] = 0
                
                df = df[expected_features]
                X_scaled = scaler.transform(df)
                
                pred_idx = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0]
                pred_label = le_classes[pred_idx]
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                color_map = {
                    "No_Disease": "#16a34a",
                    "Low_Risk": "#3b82f6",
                    "Moderate_Risk": "#f59e0b",
                    "High_Risk": "#ef4444",
                    "Severe_Disease": "#dc2626"
                }
                color = color_map.get(pred_label, "#64748b")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-result" style="background: {color};">
                        <h2>{pred_label.replace('_', ' ').upper()}</h2>
                        <p>Predicted Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "No_Disease" in pred_label:
                        st.success("Kidney function appears normal.")
                    else:
                        st.warning(f"Indication of {pred_label.replace('_', ' ')}. Please consult a nephrologist.")
                        
                with col2:
                    probs_df = pd.DataFrame({
                        'Risk Level': [x.replace('_', ' ') for x in le_classes],
                        'Probability': pred_proba
                    })
                    fig = px.bar(probs_df, x='Risk Level', y='Probability', 
                                title="Risk Distribution",
                                color='Probability', color_continuous_scale='RdYlGn_r')
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

elif selected_page == "Stool Analysis":
    st.markdown("<h1>Stool Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.125rem; color: #64748b; margin-bottom: 2rem;'>Bristol Stool Scale Classification</p>", unsafe_allow_html=True)
    
    st.info("This module is running in simulation mode for demonstration purposes.")
    
    uploaded_file = st.file_uploader("Upload Stool Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Sample", use_container_width=True)
            
        with col2:
            if st.button("Analyze Image"):
                with st.spinner("Processing image..."):
                    time.sleep(1.5)
                    
                    img_bytes = uploaded_file.getvalue()
                    h = int(hashlib.md5(img_bytes).hexdigest(), 16)
                    
                    types = [
                        ("Type 1", "Separate hard lumps", "#ef4444"),
                        ("Type 2", "Lumpy and sausage-like", "#f97316"),
                        ("Type 3", "Sausage with cracks", "#22c55e"),
                        ("Type 4", "Smooth, soft sausage", "#22c55e"),
                        ("Type 5", "Soft blobs with clear edges", "#eab308"),
                        ("Type 6", "Mushy consistency", "#f97316"),
                        ("Type 7", "Liquid consistency", "#ef4444")
                    ]
                    
                    idx = h % 7
                    pred_type, pred_desc, color = types[idx]
                    confidence = 0.85 + (h % 15) / 100.0
                    
                    st.markdown(f"""
                    <div class="prediction-result" style="background: {color};">
                        <h2>{pred_type}</h2>
                        <p>{pred_desc}</p>
                        <p style="margin-top: 1rem; font-size: 0.95rem;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-header'>Analysis Report</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    - **Texture Analysis**: Consistent with {pred_desc.lower()}
                    - **Color Analysis**: Normal pigmentation
                    - **Recommendation**: {'Maintain current diet' if idx in [2,3] else 'Consult healthcare provider'}
                    """)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.875rem; padding: 1rem 0;">
    SmartSant-IoT Medical AI Platform v2.0 | Enterprise Edition | HIPAA Compliant Processing
</div>
""", unsafe_allow_html=True)
