# SmartSant IoT - Disease Prediction System

An end-to-end AI system for predicting diseases using urine analysis and stool images.

## 🚀 Features

- **UTI Prediction**: Binary classification using urine test results
- **CKD/Proteinuria Prediction**: Binary classification using urine test results
- **Bristol Stool Classification**: Multi-class image classification (1-7)
- **Explainable AI**: SHAP values for urine model and Grad-CAM for stool model
- **RESTful API**: Easy integration with web/mobile applications

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smartsant-iot.git
   cd smartsant-iot
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

1. Prepare your data in the `data/raw` directory
2. Run the training pipeline:
   ```bash
   python3 -m training.train_models
   ```
3. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

## 📂 Project Structure

```
smart_sant_iot/
├── data/                    # Data storage
│   ├── raw/                 # Raw datasets
│   └── processed/           # Processed datasets
├── models/                  # Saved models
├── training/                # Training scripts
├── preprocessing/           # Data preprocessing
├── api/                     # FastAPI application
├── utils/                   # Utility functions
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test cases
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## 📊 Model Performance

### Urine Model (UTI & CKD Prediction)
- Accuracy: TBD
- F1 Score: TBD
- ROC-AUC: TBD

### Stool Model (Bristol Classification)
- Accuracy: TBD
- Per-class F1: TBD

## 📝 API Documentation

Once the server is running, visit `/docs` for interactive API documentation.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
