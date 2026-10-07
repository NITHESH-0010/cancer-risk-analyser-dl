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
_calibrator = None

def get_models():
    global _models, _encoder, _calibrator
    if _encoder is None:
        _encoder = load_encoder(os.path.join(base_dir, 'models', 'feature_encoder.pkl'))
    if _calibrator is None:
        try:
            _calibrator = joblib.load(os.path.join(base_dir, 'models', 'calibrator.pkl'))
        except:
            pass
            
    if 'Encoded MLP' not in _models:
        _models['Encoded MLP'] = load_dl_model(os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras'))
        
    return _models, _encoder, _calibrator

def predict_proba(raw_df, model_name='Encoded MLP'):
    models, encoder, calibrator = get_models()
    
    X_enc = encode(raw_df, encoder)
    p_raw = models['Encoded MLP'].predict(X_enc, verbose=0).flatten()
        
    if calibrator is not None:
        p_cal = calibrator.predict(p_raw)
    else:
        p_cal = p_raw
        
    p_clipped = np.clip(p_cal, 0.001, 0.999)
    return p_clipped
