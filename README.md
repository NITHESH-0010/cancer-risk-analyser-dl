# Cancer Risk Analyser

A predictive modelling pipeline and local web application for estimating cancer risk based on lifestyle and demographic factors. 

## Final Model Choice
The final model deployed in this app is an **Average Ensemble** of a tuned Gradient Boosting Classifier and a 32-unit Encoded Deep Learning Multi-Layer Perceptron (MLP). The Ensemble was chosen because it achieved the highest Test ROC-AUC (0.963) compared to either model alone (0.955 and 0.958 respectively). The models underwent robust 5-fold cross-validation on the training set and rigorous Platt scaling for calibration.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the local Flask server:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your browser.

## Performance Results

| Model | Test Accuracy | Test ROC-AUC | CV AUC (Train) | CV LogLoss |
|-------|---------------|--------------|----------------|------------|
| **Gradient Boosting** | 0.947 | 0.955 | 0.948 | 0.298 |
| **Encoded MLP** | 0.938 | 0.958 | 0.946 | 0.289 |
| **Ensemble** | 0.938 | 0.963 | 0.952 | 0.274 |

### Calibration
Probabilities are carefully calibrated using Platt scaling. Before calibration, the Ensemble Brier score was 0.0754. After calibration, the Brier score improved significantly to 0.0663, indicating high reliability.

## Limitations
- **Synthetic Data**: The model is trained on a synthetic dataset (likely generated for Kaggle) and is not based on real patient records.
- **Not Clinically Validated**: This tool has not undergone clinical trials or external validation.
- **Not Medical Advice**: The risk scores and what-if scenarios reflect mathematical patterns in the synthetic dataset and do not constitute professional medical advice, diagnosis, or treatment.
