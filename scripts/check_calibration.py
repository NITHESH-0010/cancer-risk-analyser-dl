import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from model_io import load_dl_model, load_encoder
from features import encode
from sklearn.linear_model import LogisticRegression

def check_calibration():
    # Load val data
    val_data = np.load(os.path.join(base_dir, 'data_splits', 'val_raw.npz'), allow_pickle=True)
    X_val = pd.DataFrame(val_data['X'], columns=['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
    y_val = val_data['y'].astype(int)
    
    # Load model and encoder
    model = load_dl_model(os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras'))
    encoder = load_encoder(os.path.join(base_dir, 'models', 'feature_encoder.pkl'))
    
    # Predict
    X_enc = encode(X_val, encoder)
    preds = model.predict(X_enc, verbose=0).flatten()
    
    # Metrics before
    acc_before = accuracy_score(y_val, (preds > 0.5).astype(int))
    auc_before = roc_auc_score(y_val, preds)
    brier_before = brier_score_loss(y_val, preds)
    
    print("--- Before Calibration ---")
    print(f"Accuracy: {acc_before:.4f}")
    print(f"AUC: {auc_before:.4f}")
    print(f"Brier: {brier_before:.4f}")
    
    prob_true, prob_pred = calibration_curve(y_val, preds, n_bins=10)
    print("\nReliability Table (Before):")
    print("Mean Predicted | True Fraction")
    for p, t in zip(prob_pred, prob_true):
        print(f"{p:14.4f} | {t:.4f}")
        
    # Convert probabilities to logits for Platt scaling
    # We clip to avoid log(0)
    preds_clip = np.clip(preds, 1e-6, 1 - 1e-6)
    logits = np.log(preds_clip / (1 - preds_clip))
    
    print("\nTraining Platt Scaler on Validation Set (Logits)...")
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(logits.reshape(-1, 1), y_val)
    
    cal_preds = lr.predict_proba(logits.reshape(-1, 1))[:, 1]
    
    # Metrics after
    acc_after = accuracy_score(y_val, (cal_preds > 0.5).astype(int))
    auc_after = roc_auc_score(y_val, cal_preds)
    brier_after = brier_score_loss(y_val, cal_preds)
    
    print("\n--- After Calibration ---")
    print(f"Accuracy: {acc_after:.4f}")
    print(f"AUC: {auc_after:.4f}")
    print(f"Brier: {brier_after:.4f}")
    
    prob_true_c, prob_pred_c = calibration_curve(y_val, cal_preds, n_bins=10)
    print("\nReliability Table (After):")
    print("Mean Predicted | True Fraction")
    for p, t in zip(prob_pred_c, prob_true_c):
        print(f"{p:14.4f} | {t:.4f}")
        
    # Save calibrator
    joblib.dump(lr, os.path.join(base_dir, 'models', 'platt_scaler.pkl'))
    print("\nSaved calibrator to models/platt_scaler.pkl")

if __name__ == "__main__":
    check_calibration()
