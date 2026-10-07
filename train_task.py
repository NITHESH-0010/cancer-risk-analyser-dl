import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import IsotonicRegression
from sklearn.utils.class_weight import compute_class_weight
import joblib
import random

# Ensure src is in the path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from features import ComplexFeatureEncoder, encode
from model import build_mlp
from search import CONFIGS

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    set_seeds(42)
    
    # 1. Load Data
    # We will use data_splits/train_raw.npz and val_raw.npz to get the full training set
    # Wait, the data splits were created as 70% train, 15% val, 15% test in preprocess.py
    # We can just load the raw dataframe and do the split here for CV, or use the existing train_raw
    # The prompt says "Retrain the final model on the 7 remaining features... use the same seed and settings as before"
    
    # Load dataset
    df = pd.read_csv("dataset/cancer_data.csv")
    if 'Gender' in df.columns:
        df = df.drop(columns=['Gender'])
        
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis'].values
    
    # Same splits as preprocess.py:
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    # Actually preprocess.py did:
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # Combine Train and Val for full training?
    # "Add probability calibration on a validation split". So we will train on X_train, calibrate on X_val!
    
    # Best config E: {'hidden_units': [32], 'dropout': 0.1, 'l2': 0.0, 'lr': 1e-3, 'encoded': True, 'use_class_weight': True}
    # Wait, the user said "Use the same seed and settings as before". The best config was E.
    config = CONFIGS['E'].copy()
    is_encoded = config.pop('encoded')
    use_cw = config.pop('use_class_weight')
    
    print("--- Correlation with Target ---")
    print(f"GeneticRisk correlation with target: {X_train['GeneticRisk'].corr(pd.Series(y_train, index=X_train.index)):.4f}")
    
    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # X_train is our training set. Let's do 5-fold CV on X_train to report the metrics
    cv_accs = []
    cv_aucs = []
    
    fold = 1
    for train_idx, val_idx in skf.split(X_train, y_train):
        set_seeds(42) # Fixed seed per fold to avoid randomness? Or maybe just let it vary? The prompt says "same seed and settings as before"
        
        X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        
        encoder_cv = ComplexFeatureEncoder(n_bins=8)
        encoder_cv.fit(X_tr)
        
        X_tr_enc = encoder_cv.transform(X_tr)
        X_v_enc = encoder_cv.transform(X_v)
        
        classes = np.unique(y_tr)
        weights = compute_class_weight('balanced', classes=classes, y=y_tr)
        cw = {cls: weight for cls, weight in zip(classes, weights)} if use_cw else None
        
        model_cv = build_mlp(input_dim=X_tr_enc.shape[1], **config)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        
        model_cv.fit(
            X_tr_enc, y_tr,
            validation_data=(X_v_enc, y_v),
            epochs=100,
            batch_size=32,
            class_weight=cw,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        preds = model_cv.predict(X_v_enc, verbose=0).flatten()
        acc = accuracy_score(y_v, (preds > 0.5).astype(int))
        auc = roc_auc_score(y_v, preds)
        
        cv_accs.append(acc)
        cv_aucs.append(auc)
        print(f"Fold {fold}: Accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")
        fold += 1
        
    print(f"\n--- 5-Fold CV Results ---")
    print(f"Accuracy: {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}")
    print(f"ROC-AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
    
    # Train Final Model on full X_train
    print("\n--- Training Final Model ---")
    set_seeds(42) # Wait, original train_dl.py used seeds [1, 2, 3, 4, 5] and picked the one with best val_loss.
    # The prompt says "Use the same seed and settings as before". The best seed from train_dl.py was typically 1 or whatever gave best val_loss.
    # Let's train 5 seeds and pick the best val_loss on X_val.
    
    encoder_final = ComplexFeatureEncoder(n_bins=8)
    encoder_final.fit(X_train)
    X_train_enc = encoder_final.transform(X_train)
    X_val_enc = encoder_final.transform(X_val)
    X_test_enc = encoder_final.transform(X_test)
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = {cls: weight for cls, weight in zip(classes, weights)} if use_cw else None
    
    best_val_loss = float('inf')
    best_model = None
    
    for seed in [1, 2, 3, 4, 5]:
        set_seeds(seed)
        tf.keras.backend.clear_session()
        
        model = build_mlp(input_dim=X_train_enc.shape[1], **config)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        
        history = model.fit(
            X_train_enc, y_train,
            validation_data=(X_val_enc, y_val),
            epochs=100,
            batch_size=32,
            class_weight=cw,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            
    # Calibration on Validation Split
    print("\n--- Calibration ---")
    uncalibrated_val_preds = best_model.predict(X_val_enc, verbose=0).flatten()
    brier_before = brier_score_loss(y_val, uncalibrated_val_preds)
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(uncalibrated_val_preds, y_val)
    
    calibrated_val_preds = iso.predict(uncalibrated_val_preds)
    brier_after = brier_score_loss(y_val, calibrated_val_preds)
    
    print(f"Brier Score Before: {brier_before:.4f}")
    print(f"Brier Score After:  {brier_after:.4f}")
    
    # Overwrite the saved models
    print("\nSaving final model and encoder...")
    os.makedirs('models', exist_ok=True)
    best_model.save('models/cancer_dl_model_v2.keras')
    joblib.dump(encoder_final, 'models/feature_encoder.pkl')
    joblib.dump(iso, 'models/calibrator.pkl')
    
    # Check max probability to ensure no profile shows exactly 100%
    # We will wrap the prediction logic in a new function that clips max to 0.999
    
    # Feature Importance via Permutation on Test Set
    print("\n--- Permutation Importance on Test Set (20 repeats) ---")
    set_seeds(42)
    test_preds_uncal = best_model.predict(X_test_enc, verbose=0).flatten()
    test_preds = iso.predict(test_preds_uncal)
    
    base_auc = roc_auc_score(y_test, test_preds)
    
    importances = {col: [] for col in X_test.columns}
    
    for repeat in range(20):
        for col in X_test.columns:
            X_test_permuted = X_test.copy()
            X_test_permuted[col] = np.random.permutation(X_test_permuted[col].values)
            
            X_test_permuted_enc = encoder_final.transform(X_test_permuted)
            preds_uncal = best_model.predict(X_test_permuted_enc, verbose=0).flatten()
            preds_cal = iso.predict(preds_uncal)
            
            perm_auc = roc_auc_score(y_test, preds_cal)
            importances[col].append(base_auc - perm_auc)
            
    imp_mean = {col: np.mean(vals) for col, vals in importances.items()}
    imp_std = {col: np.std(vals) for col, vals in importances.items()}
    
    # Ranked table
    print("\nFeature | Mean AUC Drop | Std")
    print("-" * 40)
    for col in sorted(imp_mean, key=imp_mean.get, reverse=True):
        print(f"{col:15} | {imp_mean[col]:.4f} | {imp_std[col]:.4f}")
        
    print("\nDone. Please review the output above.")

if __name__ == "__main__":
    main()
