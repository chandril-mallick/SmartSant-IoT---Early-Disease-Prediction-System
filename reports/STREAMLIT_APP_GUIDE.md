# 🌐 SmartSant-IoT Web Application

## Streamlit Visual Classification Interface

A beautiful, interactive web application for disease prediction with visual analytics.

### 🎯 Features

- **💧 Urine Analysis**: Interactive form for UTI detection with real-time probability gauges
- **🫘 Kidney Disease**: 5-level CKD risk assessment (coming soon)
- **🔬 Stool Analysis**: Bristol Stool Scale image classification (coming soon)
- **📊 Visual Analytics**: Gauge charts, metrics, and interactive visualizations
- **🎨 Modern UI**: Gradient designs, responsive layout, professional styling

### 🚀 Quick Start

#### Method 1: Using the Launch Script (Recommended)
```bash
./run_app.sh
```

#### Method 2: Manual Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install streamlit plotly

# Run the app
streamlit run app.py
```

### 📱 Access the Application

Once running, open your browser and navigate to:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

### 🎨 User Interface

#### Home Page
- System overview and statistics
- Performance metrics
- Quick navigation to prediction modules

#### Urine Analysis Page
- 15-parameter input form
- Real-time UTI probability calculation
- Visual gauge charts
- Risk factor analysis
- Clinical recommendations

#### Kidney Disease Page (Coming Soon)
- 57 clinical feature inputs
- 5-level risk classification
- Comprehensive health assessment

#### Stool Analysis Page (Coming Soon)
- Image upload interface
- Bristol Scale classification (Types 1-7)
- Grad-CAM visualizations
- Health insights

### 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML Backend**: Scikit-learn, PyTorch
- **Styling**: Custom CSS with gradient themes

### 📊 Urine Analysis Features

The urine analysis module includes:

**Input Parameters**:
1. Leukocyte Esterase (0-3)
2. Nitrite (0-1)
3. Protein (0-3)
4. Blood (0-3)
5. Glucose (0-3)
6. Ketones (0-3)
7. WBC Count (cells/μL)
8. RBC Count (cells/μL)
9. Bacteria Count (0-4)
10. pH (4.5-8.5)
11. Specific Gravity (1.000-1.030)
12. Creatinine (mg/dL)
13. Turbidity (0-3)
14. Conductivity (mS/cm)
15. Patient Age

**Output**:
- UTI Detection (Yes/No)
- Probability Gauge (0-100%)
- Risk Factor Analysis
- Clinical Recommendations

### 🎯 Model Performance Display

The app shows real-time model performance:
- **Accuracy**: 93.06%
- **Precision**: 38.89%
- **Recall**: 43.75%
- **F1-Score**: 0.4118
- **AUC-ROC**: 0.7053

### 🔧 Configuration

Edit `app.py` to customize:
- Color schemes (CSS gradients)
- Model paths
- Threshold values
- UI layout

### 📝 Usage Example

1. **Launch the app**: `./run_app.sh`
2. **Select "Urine Analysis"** from sidebar
3. **Enter test parameters** in the form
4. **Click "Analyze Urine Sample"**
5. **View results** with probability gauge and risk factors

### 🎨 Customization

#### Change Color Theme
Edit the CSS in `app.py`:
```python
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

#### Modify Gauge Thresholds
Update the gauge chart ranges:
```python
'steps': [
    {'range': [0, 30], 'color': '#90EE90'},   # Low risk (green)
    {'range': [30, 70], 'color': '#FFD700'},  # Medium risk (yellow)
    {'range': [70, 100], 'color': '#FF6B6B'}  # High risk (red)
]
```

### 🚀 Deployment

#### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

#### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

#### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 📱 Mobile Responsive

The app is fully responsive and works on:
- Desktop browsers
- Tablets
- Mobile phones

### 🔒 Security Notes

- Models run locally (no data sent to external servers)
- No patient data is stored
- HIPAA-compliant design
- All processing happens in-memory

### 🐛 Troubleshooting

**Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

**Module not found**:
```bash
pip install -r requirements.txt
```

**Model not found**:
```bash
python3 training/optimize_urine_classifier.py
```

### 📚 Documentation

For more details, see:
- [Main README](README.md)
- [Urine Classifier Report](URINE_CLASSIFIER_OPTIMIZATION_REPORT.md)
- [Kidney Preprocessing Guide](KIDNEY_PREPROCESSING_GUIDE.md)
- [Stool Model Evaluation](STOOL_MODEL_EVALUATION_REPORT.md)

### 🤝 Contributing

To add new features to the web app:
1. Edit `app.py`
2. Test locally
3. Update this README
4. Submit a pull request

### 📞 Support

For issues with the web app:
- Check console for error messages
- Verify model files exist
- Ensure all dependencies are installed
- Check Streamlit documentation

---

**Built with ❤️ using Streamlit**

*Last updated: 2025-11-25*
