import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from features import encode, ComplexFeatureEncoder
from model_io import load_dl_model, load_encoder

def build_mlp_model(input_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.0),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_oof_predictions(X, y, X_encoded, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_gb = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))
    
    # Simple grid for GB
    grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }
    
    best_gb_params = None
    best_gb_auc = -1
    
    print("Tuning Gradient Boosting...")
    for params in ParameterGrid(grid):
        aucs = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            gb = GradientBoostingClassifier(random_state=42, **params)
            gb.fit(X_tr, y_tr)
            preds = gb.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, preds))
        mean_auc = np.mean(aucs)
        if mean_auc > best_gb_auc:
            best_gb_auc = mean_auc
            best_gb_params = params
            
    print(f"Best GB params: {best_gb_params} with AUC {best_gb_auc:.4f}")
    
    # Now get OOF for Best GB and MLP
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        X_enc_tr, X_enc_va = X_encoded[train_idx], X_encoded[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        # GB
        gb = GradientBoostingClassifier(random_state=42, **best_gb_params)
        gb.fit(X_tr, y_tr)
        oof_gb[val_idx] = gb.predict_proba(X_va)[:, 1]
        
        # MLP
        mlp = build_mlp_model(X_enc_tr.shape[1])
        # Simple class weight handling if needed, but let's use default for CV
        class_weight = {0: 1.0, 1: (len(y_tr)-sum(y_tr))/sum(y_tr)}
        mlp.fit(X_enc_tr, y_tr, epochs=30, batch_size=32, verbose=0, class_weight=class_weight)
        oof_mlp[val_idx] = mlp.predict(X_enc_va, verbose=0).flatten()
        
    return oof_gb, oof_mlp, best_gb_params

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load splits
    train_raw = np.load('data_splits/train_raw.npz')
    X_train, y_train = train_raw['X'], train_raw['y']
    
    val_raw = np.load('data_splits/val_raw.npz')
    X_val, y_val = val_raw['X'], val_raw['y']
    
    test_raw = np.load('data_splits/test_raw.npz')
    X_test, y_test = test_raw['X'], test_raw['y']
    
    encoder = load_encoder('models/feature_encoder.pkl')
    X_train_enc = encode(X_train, encoder)
    X_val_enc = encode(X_val, encoder)
    X_test_enc = encode(X_test, encoder)
    
    oof_gb, oof_mlp, gb_params = get_oof_predictions(X_train, y_train, X_train_enc)
    oof_ens = (oof_gb + oof_mlp) / 2.0
    
    print("\n--- Out-Of-Fold (Train) CV Metrics ---")
    models_oof = {'Gradient Boosting': oof_gb, 'Encoded MLP': oof_mlp, 'Ensemble': oof_ens}
    cv_metrics = []
    for name, preds in models_oof.items():
        auc = roc_auc_score(y_train, preds)
        loss = log_loss(y_train, preds)
        cv_metrics.append({'Model': name, 'CV_AUC': auc, 'CV_LogLoss': loss})
        print(f"{name}: AUC={auc:.4f}, Log-Loss={loss:.4f}")
        
    # Fit full models on Train
    gb_full = GradientBoostingClassifier(random_state=42, **gb_params)
    gb_full.fit(X_train, y_train)
    
    # Existing MLP
    mlp_full = load_dl_model('models/cancer_dl_model_v2.keras')
    
    # Predict on Validation
    val_gb = gb_full.predict_proba(X_val)[:, 1]
    val_mlp = mlp_full.predict(X_val_enc, verbose=0).flatten()
    val_ens = (val_gb + val_mlp) / 2.0
    
    print("\n--- Validation Metrics ---")
    models_val = {'Gradient Boosting': val_gb, 'Encoded MLP': val_mlp, 'Ensemble': val_ens}
    best_model_name = None
    best_val_auc = -1
    best_val_loss = float('inf')
    
    for name, preds in models_val.items():
        auc = roc_auc_score(y_val, preds)
        loss = log_loss(y_val, preds)
        print(f"{name}: AUC={auc:.4f}, Log-Loss={loss:.4f}")
        if auc > best_val_auc or (auc == best_val_auc and loss < best_val_loss):
            best_val_auc = auc
            best_val_loss = loss
            best_model_name = name
            
    print(f"\nWinning Model by Validation: {best_model_name}")
    
    # Calibrate models using OOF predictions (Platt Scaling)
    print("\n--- Calibration ---")
    from sklearn.linear_model import LogisticRegression
    calibrators = {}
    
    plt.figure(figsize=(8, 6))
    
    for name, oof_preds in models_oof.items():
        brier_before = brier_score_loss(y_train, oof_preds)
        loss_before = log_loss(y_train, oof_preds)
        
        # Train calibrator
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(oof_preds.reshape(-1, 1), y_train)
        calibrators[name] = lr
        
        cal_preds = lr.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        brier_after = brier_score_loss(y_train, cal_preds)
        loss_after = log_loss(y_train, cal_preds)
        
        print(f"{name}: Brier before={brier_before:.4f}, after={brier_after:.4f} | LogLoss before={loss_before:.4f}, after={loss_after:.4f}")
        
        prob_true, prob_pred = calibration_curve(y_train, cal_preds, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Curve (Platt Scaling)')
    plt.legend()
    plt.savefig('results/calibration.png')
    
    # Save calibrators and GB model
    joblib.dump(gb_full, 'models/final_gb.pkl')
    joblib.dump(calibrators, 'models/calibrators.pkl')
    
    # Evaluate EXACTLY ONCE on Test Split
    print("\n--- Test Split Metrics (EXACTLY ONCE) ---")
    
    # Generate test predictions
    test_gb = gb_full.predict_proba(X_test)[:, 1]
    test_mlp = mlp_full.predict(X_test_enc, verbose=0).flatten()
    test_ens = (test_gb + test_mlp) / 2.0
    
    models_test = {'Gradient Boosting': test_gb, 'Encoded MLP': test_mlp, 'Ensemble': test_ens}
    
    final_metrics = []
    for name, preds in models_test.items():
        # Calibrate
        lr = calibrators[name]
        cal_preds = lr.predict_proba(preds.reshape(-1, 1))[:, 1]
        cal_preds = np.clip(cal_preds, 0.001, 0.999)
        
        bin_preds = (cal_preds >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, bin_preds)
        prec = precision_score(y_test, bin_preds)
        rec = recall_score(y_test, bin_preds)
        f1 = f1_score(y_test, bin_preds)
        auc = roc_auc_score(y_test, cal_preds)
        brier = brier_score_loss(y_test, cal_preds)
        
        final_metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': auc,
            'Brier': brier
        })
        
    final_df = pd.DataFrame(final_metrics)
    final_df.to_csv('results/final_metrics.csv', index=False)
    print(final_df.to_string(index=False))

if __name__ == '__main__':
    main()
