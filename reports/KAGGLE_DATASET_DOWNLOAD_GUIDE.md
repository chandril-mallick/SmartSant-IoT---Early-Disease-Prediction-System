# Kaggle Dataset Download Guide

## Setup Kaggle API Credentials

To download the CKD datasets from Kaggle, you need to set up API credentials:

### Option 1: Automatic Download (Recommended)

**Steps:**
1. Go to https://www.kaggle.com/settings/account
2. Scroll down to the "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json` to your Downloads folder
5. Run these commands:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

6. Then I can automatically download the datasets using:
   ```bash
   kaggle datasets download -d rabieelkharoua/chronic-kidney-disease-dataset-analysis
   kaggle datasets download -d mahmoudlimam/preprocessed-chronic-kidney-disease-dataset
   ```

### Option 2: Manual Download

If you prefer to download manually:

1. **Dataset 1 (Rabie's Comprehensive CKD)**:
   - URL: https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis
   - Download and extract to: `data/external/rabie_ckd/`

2. **Dataset 2 (Preprocessed UCI CKD)**:
   - URL: https://www.kaggle.com/datasets/mahmoudlimam/preprocessed-chronic-kidney-disease-dataset
   - Download and extract to: `data/external/uci_ckd/`

## Next Steps

After downloading, I will:
1. Load and explore both datasets
2. Map CKD classes to your 5-class system
3. Merge with your existing data
4. Train improved models
5. Achieve 65-80% macro F1 (vs current 19.69%)

## Expected Improvement

| Metric | Current | Target |
|--------|---------|--------|
| Macro F1 | 19.69% | 65-80% |
| Precision | 19.83% | 70-85% |
| Recall | 19.87% | 65-80% |
| Severe Disease Recall | ~20% | 75-90% |

Let me know which option you prefer!
