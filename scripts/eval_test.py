import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from model_io import load_dl_model, load_encoder
from features import encode
from explain_one import get_true_logit

def eval_test():
    test_data = np.load(os.path.join(base_dir, 'data_splits', 'test_raw.npz'), allow_pickle=True)
    X_test = pd.DataFrame(test_data['X'], columns=['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
    y_test = test_data['y'].astype(int)
    
    model = load_dl_model(os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras'))
    encoder = load_encoder(os.path.join(base_dir, 'models', 'feature_encoder.pkl'))
    calibrator = joblib.load(os.path.join(base_dir, 'models', 'platt_scaler.pkl'))
    
    encoded = encode(X_test, encoder)
    
    cal_preds = []
    for i in range(len(X_test)):
        df_row = X_test.iloc[[i]]
        logit = get_true_logit(df_row, model, encoder)
        p = calibrator.predict_proba([[logit]])[0][1]
        cal_preds.append(p)
    cal_preds = np.array(cal_preds)
    
    acc = accuracy_score(y_test, (cal_preds > 0.5).astype(int))
    auc = roc_auc_score(y_test, cal_preds)
    brier = brier_score_loss(y_test, cal_preds)
    
    print(f"Test Accuracy: {acc*100:.1f}%")
    print(f"Test AUC: {auc:.3f}")
    print(f"Test Brier: {brier:.4f}")

if __name__ == "__main__":
    eval_test()
