import os
import sys
import numpy as np
import pandas as pd
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from features import encode
from model_io import load_encoder, load_dl_model

# Cache models
_models = {}
_encoder = None
_calibrators = None

def get_models():
    global _models, _encoder, _calibrators
    if _encoder is None:
        _encoder = load_encoder(os.path.join(base_dir, 'models', 'feature_encoder.pkl'))
    if _calibrators is None:
        _calibrators = joblib.load(os.path.join(base_dir, 'models', 'calibrators.pkl'))
        
    if 'Gradient Boosting' not in _models:
        _models['Gradient Boosting'] = joblib.load(os.path.join(base_dir, 'models', 'final_gb.pkl'))
    if 'Encoded MLP' not in _models:
        _models['Encoded MLP'] = load_dl_model(os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras'))
        
    return _models, _encoder, _calibrators

def predict_proba(raw_df, model_name='Ensemble'):
    models, encoder, calibrators = get_models()
    
    if model_name in ['Encoded MLP', 'Ensemble']:
        X_enc = encode(raw_df, encoder)
        p_mlp = models['Encoded MLP'].predict(X_enc, verbose=0).flatten()
        
    if model_name in ['Gradient Boosting', 'Ensemble']:
        # Ensure correct order
        expected_cols = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
        X_raw = raw_df[expected_cols].values
        p_gb = models['Gradient Boosting'].predict_proba(X_raw)[:, 1]
        
    if model_name == 'Ensemble':
        p_raw = (p_mlp + p_gb) / 2.0
    elif model_name == 'Encoded MLP':
        p_raw = p_mlp
    elif model_name == 'Gradient Boosting':
        p_raw = p_gb
    else:
        raise ValueError(f"Unknown model_name {model_name}")
        
    # Calibrate
    calibrator = calibrators[model_name]
    p_cal = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
    
    # Clip
    p_clipped = np.clip(p_cal, 0.001, 0.999)
    return p_clipped
