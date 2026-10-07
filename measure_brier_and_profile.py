import os
import numpy as np
import pandas as pd
import joblib

import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from model_io import load_dl_model, load_encoder
from features import encode
from sklearn.metrics import brier_score_loss

def measure_test_brier():
    print("\n=== Test Split Brier Scores ===")
    test_raw = np.load('data_splits/test_raw.npz')
    X_test, y_test = test_raw['X'], test_raw['y']
    
    gb_full = joblib.load('models/final_gb.pkl')
    mlp_full = load_dl_model('models/cancer_dl_model_v2.keras')
    calibrators = joblib.load('models/calibrators.pkl')
    encoder = load_encoder('models/feature_encoder.pkl')
    
    X_test_enc = encode(X_test, encoder)
    
    test_gb = gb_full.predict_proba(X_test)[:, 1]
    test_mlp = mlp_full.predict(X_test_enc, verbose=0).flatten()
    test_ens = (test_gb + test_mlp) / 2.0
    
    models_test = {'Gradient Boosting': test_gb, 'Encoded MLP': test_mlp, 'Ensemble': test_ens}
    
    for name, preds in models_test.items():
        brier_before = brier_score_loss(y_test, preds)
        cal_preds = calibrators[name].predict_proba(preds.reshape(-1, 1))[:, 1]
        brier_after = brier_score_loss(y_test, cal_preds)
        print(f"{name} (TEST split): Before={brier_before:.4f}, After={brier_after:.4f}")

def age_profiling():
    print("\n=== Age Profiling (Ensemble) ===")
    gb_full = joblib.load('models/final_gb.pkl')
    mlp_full = load_dl_model('models/cancer_dl_model_v2.keras')
    calibrators = joblib.load('models/calibrators.pkl')
    encoder = load_encoder('models/feature_encoder.pkl')
    
    # We create a dataframe with 10 ages, and (CancerHistory x Smoking)
    ages = np.linspace(20, 80, 10)
    
    # Defaults
    bmi = 25
    gen_risk = 0
    phys = 5
    alc = 2
    
    for ch in [0, 1]:
        for sm in [0, 1]:
            print(f"\nCancerHistory={ch}, Smoking={sm}:")
            rows = []
            for a in ages:
                rows.append({
                    'Age': a,
                    'BMI': bmi,
                    'Smoking': sm,
                    'GeneticRisk': gen_risk,
                    'PhysicalActivity': phys,
                    'AlcoholIntake': alc,
                    'CancerHistory': ch
                })
            df = pd.DataFrame(rows)
            X_enc = encode(df, encoder)
            
            gb_preds = gb_full.predict_proba(df)[:, 1]
            mlp_preds = mlp_full.predict(X_enc, verbose=0).flatten()
            ens_preds = (gb_preds + mlp_preds) / 2.0
            
            cal_ens = calibrators['Ensemble'].predict_proba(ens_preds.reshape(-1, 1))[:, 1]
            
            for i, a in enumerate(ages):
                print(f"Age {a:.1f} -> {cal_ens[i]:.4f}")

if __name__ == '__main__':
    measure_test_brier()
    age_profiling()
