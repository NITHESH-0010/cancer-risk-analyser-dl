import joblib
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('src'))
from features import ComplexFeatureEncoder

def debug():
    encoder = joblib.load('models/feature_encoder.pkl')
    train_data = np.load('data_splits/train_raw.npz')
    X_train_raw = train_data['X']
    
    X_enc_array = encoder.transform(X_train_raw)
    
    print("--- Encoder Output Shape ---")
    print(X_enc_array.shape)
    
    print("\n--- Standard Deviation of Encoded Columns ---")
    stds = X_enc_array.std(axis=0)
    for i, std in enumerate(stds):
        print(f"Col {i}: std = {std:.4f}")
        if std == 0.0:
            print(f"  -> BUG: Column {i} has std 0.0!")
            
    print("\n--- Sample Rows (Raw vs Encoded) ---")
    for i in range(5):
        print(f"Row {i} Raw:")
        print(X_train_raw[i])
        print(f"Row {i} Enc:")
        print(X_enc_array[i])
        print()
        
    try:
        from tensorflow.keras.models import load_model
        model = load_model('models/cancer_dl_model_v2.keras')
        print(f"\nModel v2 expected input shape: {model.input_shape}")
    except Exception as e:
        print(f"\nCould not load model: {e}")

if __name__ == '__main__':
    debug()
