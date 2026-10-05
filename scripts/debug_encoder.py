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
    
    # Reconstruct feature names
    # 1. One-hot
    oh_names = encoder.onehot.get_feature_names_out().tolist()
    # 2. Piecewise
    pw_names = []
    for col in encoder.piecewise_cols:
        for i in range(encoder.piecewise.n_bins):
            pw_names.append(f"{col}_bin_{i}")
    # 3. Passthrough
    pt_names = encoder.passthrough_cols
    
    feature_names = oh_names + pw_names + pt_names
    
    X_enc_array = encoder.transform(X_train_raw)
    
    print("--- Encoded Feature Names/Order ---")
    for i, name in enumerate(feature_names):
        print(f"Col {i}: {name}")
        
    print("\n--- Encoder Output Shape ---")
    print(X_enc_array.shape)
    
    print("\n--- Standard Deviation of Encoded Columns ---")
    stds = X_enc_array.std(axis=0)
    for i, std in enumerate(stds):
        print(f"Col {i}: std = {std:.4f}")
        if std == 0.0:
            print(f"  -> BUG: Column {i} has std 0.0!")
            
    print("\n--- Sample Rows (Raw vs Encoded) ---")
    # For printing raw vs encoded properly
    raw_cols = encoder.raw_columns
    for i in range(5):
        print(f"Row {i} Raw:")
        for j, val in enumerate(X_train_raw[i]):
            print(f"  {raw_cols[j]}: {val}")
        print(f"Row {i} Enc:")
        for j, val in enumerate(X_enc_array[i]):
            print(f"  {feature_names[j]}: {val:.4f}")
        print()
        
    try:
        from tensorflow.keras.models import load_model
        model = load_model('models/cancer_dl_model_v2.keras')
        print(f"\nModel v2 expected input shape: {model.input_shape}")
    except Exception as e:
        print(f"\nCould not load model: {e}")

if __name__ == '__main__':
    debug()
