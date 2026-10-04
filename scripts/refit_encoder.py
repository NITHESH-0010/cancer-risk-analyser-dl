import sys
import os
import joblib
import numpy as np

sys.path.insert(0, 'src')
from features import ComplexFeatureEncoder

def refit_encoder():
    train_raw = np.load('data_splits/train_raw.npz')
    X_train_raw = train_raw['X']
    
    encoder = ComplexFeatureEncoder(n_bins=8)
    encoder.fit(X_train_raw)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/feature_encoder.pkl')
    print("Fitted and saved ComplexFeatureEncoder properly.")

if __name__ == "__main__":
    refit_encoder()
