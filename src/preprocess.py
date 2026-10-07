import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import load_and_verify_data

def preprocess_and_split():
    # Load data
    df = load_and_verify_data()
    
    # Define features and target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    # 70/15/15 stratified split
    # First split to 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # Then split temp to 50% val, 50% test (which is 15% / 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Separate numeric and categorical/binary columns for scaling
    # Typically, we don't scale binary columns like CancerHistory
    # Let's identify numeric columns that have >2 unique values
    numeric_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]
    binary_cols = [col for col in X_train.columns if X_train[col].nunique() <= 2]
    
    print(f"Scaling numeric columns: {numeric_cols}")
    print(f"Not scaling binary columns: {binary_cols}")
    
    # Fit StandardScaler on TRAIN set only
    scaler = StandardScaler()
    
    # Create copies to avoid SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/dl_scaler.pkl')
    print("Saved scaler to models/dl_scaler.pkl")
    
    # Save the splits as .npz
    os.makedirs('data_splits', exist_ok=True)
    
    np.savez('data_splits/train.npz', X=X_train_scaled.values, y=y_train.values)
    np.savez('data_splits/val.npz', X=X_val_scaled.values, y=y_val.values)
    np.savez('data_splits/test.npz', X=X_test_scaled.values, y=y_test.values)
    
    # Also save column names just in case they're needed later
    joblib.dump(X_train.columns.tolist(), 'data_splits/feature_names.pkl')
    
    print("Saved data splits to data_splits/")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

if __name__ == "__main__":
    preprocess_and_split()
