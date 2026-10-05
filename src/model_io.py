import os
import joblib
import sys

# Ensure src is in the path so joblib can find features module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.models import load_model

def load_encoder(path='models/feature_encoder.pkl'):
    return joblib.load(path)

def load_dl_model(path='models/cancer_dl_model_v2.keras'):
    return load_model(path)
