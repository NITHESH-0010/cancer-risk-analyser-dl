# Cancer Risk Analyser DL

This project is an end-to-end Machine Learning pipeline that predicts cancer risk based on lifestyle and health factors (Age, Gender, BMI, Smoking, Genetic Risk, Physical Activity, Alcohol Intake, Cancer History).

The project features a **Deep Learning Multi-Layer Perceptron (MLP)** that utilizes a specialized `ComplexFeatureEncoder` (one-hot encoding for categorical variables and quantile piecewise linear encoding for continuous variables) to explicitly model the highly non-linear, step-like relationships found in the dataset. A Streamlit application is provided for interactive risk assessment.

## How to Run the App

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Performance Results

After a robust grid search and 5-seed retraining, the encoded MLP demonstrates strong performance that approaches that of tree-based models on both the held-out test split and via 5-Fold Cross-Validation.

### Test Split Results (5-seed average for MLPs)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| MLP raw features | 0.8569 +/- 0.0119 | 0.8097 +/- 0.0098 | 0.8000 +/- 0.0326 | 0.8046 +/- 0.0194 | 0.9257 +/- 0.0036 |
| MLP encoded features | 0.9129 +/- 0.0132 | 0.9015 +/- 0.0233 | 0.8578 +/- 0.0132 | 0.8791 +/- 0.0177 | 0.9498 +/- 0.0050 |
| Logistic Regression | 0.8311 | 0.8000 | 0.7229 | 0.7595 | 0.9057 |
| Random Forest | 0.9289 | 0.9589 | 0.8434 | 0.8974 | 0.9474 |
| Gradient Boosting | 0.9422 | 0.9487 | 0.8916 | 0.9193 | 0.9570 |

### 5-Fold Cross-Validation Results

| Model | Accuracy (CV) | ROC-AUC (CV) |
| --- | --- | --- |
| Logistic Regression | 0.8473 +/- 0.0183 | 0.9179 +/- 0.0134 |
| Random Forest | 0.9227 +/- 0.0071 | 0.9508 +/- 0.0113 |
| Gradient Boosting | 0.9307 +/- 0.0083 | 0.9531 +/- 0.0119 |
| MLP (DL) | 0.8820 +/- 0.0105 | 0.9391 +/- 0.0133 |
| MLP encoded features | 0.9120 +/- 0.0117 | 0.9510 +/- 0.0095 |

## Limitations and Disclaimer
- **Not for Clinical Use:** This model is trained on a likely synthetic Kaggle dataset and has not been externally or clinically validated.
- **Medical Disclaimer:** The predictions provided by this tool are strictly for educational and demonstration purposes. Do not use this tool for medical advice, diagnosis, or treatment decisions.
