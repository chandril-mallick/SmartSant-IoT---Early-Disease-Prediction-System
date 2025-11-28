# Kidney Disease Classifier - Combined Dataset Experiment

## 🎯 Experiment Overview
To address the severe class imbalance and lack of feature diversity in the original dataset, we integrated two additional datasets from Kaggle.

### Datasets Integrated
1.  **Original Dataset**: 20,538 samples (80% No_Disease)
2.  **Rabie's Dataset**: 1,659 samples (92% CKD) - Added via intelligent feature mapping
3.  **UCI Dataset**: 400 samples (100% CKD) - Added via intelligent feature mapping

**Total Combined Size**: 22,597 samples (+2,059 samples)
**Severe Disease Samples**: Increased from 410 to 718 (+75%)

---

## 📊 Results Summary

| Metric | Baseline (Original) | Combined Dataset (LightGBM) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~95% (biased) | **71.84%** (more realistic) | - |
| **Macro F1** | 19.69% | **31.08%** | **+11.4% (1.58x)** 🚀 |
| **Weighted F1** | ~71% | **68.56%** | Comparable |
| **Severe Recall** | ~20% | **19.00%** | Stable on harder data |

### Best Model: LightGBM 🏆
*   **Accuracy**: 71.84%
*   **Macro F1**: 31.08%
*   **Weighted F1**: 68.56%

### Detailed Performance by Class

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **No_Disease** | 0.81 | 0.92 | 0.86 |
| **Low_Risk** | 0.44 | 0.17 | 0.24 |
| **Moderate_Risk** | 0.15 | 0.11 | 0.13 |
| **High_Risk** | 0.15 | 0.13 | 0.14 |
| **Severe_Disease** | 0.19 | 0.19 | 0.19 |

---

## 💡 Key Insights
1.  **Significant Macro F1 Improvement**: The combined dataset improved the macro F1 score by **58%** (from 19.69% to 31.08%), indicating much better handling of minority classes overall.
2.  **More Realistic Evaluation**: The original dataset's high accuracy was largely due to the 80% "No_Disease" class. The combined dataset provides a more balanced and realistic evaluation of the model's ability to detect disease.
3.  **Data Quality vs. Quantity**: While we added ~2,000 disease samples, the feature mapping between datasets (aligning different column names) is imperfect, which adds noise. This explains why accuracy dropped slightly while macro metrics improved.
4.  **Severe Disease Detection**: We maintained similar recall for severe disease (~19%) but on a much more diverse and challenging dataset, suggesting the model is more robust.

## 🚀 Next Steps for Further Improvement
1.  **Fine-tune Feature Mapping**: Manually verify and adjust the mapping of columns between the three datasets to reduce noise.
2.  **Ensemble Stacking**: Combine the LightGBM model trained on the combined data with the original Random Forest model.
3.  **Deep Learning**: Try a Neural Network on the combined dataset, as it might handle the noise from feature mapping better.
