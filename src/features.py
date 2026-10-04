import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from data import load_and_verify_data

# Piecewise linear encoder based on quantiles helps neural networks capture step-like non-linearities smoothly
class QuantilePiecewiseLinearEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=8):
        self.n_bins = n_bins
        self.quantiles_ = {}
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            # Calculate quantiles for each column
            q = np.linspace(0, 1, self.n_bins + 1)
            self.quantiles_[col] = np.percentile(X_df[col].dropna(), q * 100)
            # Ensure unique bin edges by adding a tiny noise if needed, though exact duplicates can be dropped
            self.quantiles_[col] = np.unique(self.quantiles_[col])
            # If after unique we have less than 2 edges, fallback to min/max
            if len(self.quantiles_[col]) < 2:
                self.quantiles_[col] = np.array([X_df[col].min(), X_df[col].max()])
        return self
        
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        out = []
        for col in X_df.columns:
            edges = self.quantiles_[col]
            n_intervals = len(edges) - 1
            col_encoded = np.zeros((len(X_df), n_intervals))
            
            for i in range(n_intervals):
                lower = edges[i]
                upper = edges[i+1]
                val = X_df[col].values
                
                # 0 below bin, 1 above bin, linear inside
                if upper > lower:
                    fraction = (val - lower) / (upper - lower)
                else:
                    fraction = np.zeros_like(val) # Should not happen with unique edges
                    
                clipped = np.clip(fraction, 0, 1)
                col_encoded[:, i] = clipped
                
            out.append(col_encoded)
        return np.hstack(out)


class ComplexFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=8):
        self.n_bins = n_bins
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.piecewise = QuantilePiecewiseLinearEncoder(n_bins=self.n_bins)
        
        # Define column roles
        self.onehot_col = 'GeneticRisk'
        self.piecewise_cols = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
        self.passthrough_cols = ['Gender', 'Smoking', 'CancerHistory']
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.onehot_col.split() + self.piecewise_cols + self.passthrough_cols if isinstance(X, np.ndarray) else X.columns)
        
        # Fit onehot
        self.onehot.fit(X_df[[self.onehot_col]])
        
        # Fit piecewise
        self.piecewise.fit(X_df[self.piecewise_cols])
        
        # Transform to fit scaler
        X_transformed = self._transform_without_scale(X_df)
        self.scaler.fit(X_transformed)
        return self
        
    def _transform_without_scale(self, X_df):
        # 1. One-hot
        oh_encoded = self.onehot.transform(X_df[[self.onehot_col]])
        
        # 2. Piecewise
        pw_encoded = self.piecewise.transform(X_df[self.piecewise_cols])
        
        # 3. Passthrough
        pt_encoded = X_df[self.passthrough_cols].values
        
        return np.hstack([oh_encoded, pw_encoded, pt_encoded])
        
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.onehot_col.split() + self.piecewise_cols + self.passthrough_cols if isinstance(X, np.ndarray) else X.columns)
        X_transformed = self._transform_without_scale(X_df)
        return self.scaler.transform(X_transformed)


def regenerate_raw_splits_and_encode():
    # The previous .npz files only held scaled data, so we recreate the raw splits with the same indices.
    # We use the exact same split logic from src/preprocess.py
    df = load_and_verify_data()
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print("Re-created raw splits with original indices since old .npz only held scaled data.")
    
    # Save raw splits
    os.makedirs('data_splits', exist_ok=True)
    np.savez('data_splits/train_raw.npz', X=X_train_raw.values, y=y_train.values)
    np.savez('data_splits/val_raw.npz', X=X_val_raw.values, y=y_val.values)
    np.savez('data_splits/test_raw.npz', X=X_test_raw.values, y=y_test.values)
    
    # Fit encoder
    encoder = ComplexFeatureEncoder(n_bins=8)
    # Important: Fit on TRAIN data only
    encoder.fit(X_train_raw)
    
    # Save encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/feature_encoder.pkl')
    print("Fitted and saved ComplexFeatureEncoder to models/feature_encoder.pkl")
    
    # Note: We do not save encoded splits here. We will encode on the fly or just use the encoder in subsequent scripts.

if __name__ == "__main__":
    regenerate_raw_splits_and_encode()
