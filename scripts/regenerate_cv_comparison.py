import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from model import build_mlp
from features import ComplexFeatureEncoder, encode
from search import CONFIGS, set_seeds
from data import load_and_verify_data

def run_cv():
    df = load_and_verify_data()
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    
    # 1. Scikit-learn models (unencoded, scaled)
    for name, model in models.items():
        accs, aucs = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            accs.append(accuracy_score(y_test, preds))
            aucs.append(roc_auc_score(y_test, probs))
            
        results.append({
            'Model': name,
            'Accuracy (CV)': f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
            'ROC-AUC (CV)': f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
        })
        
    # 2. MLP (DL) unencoded
    accs, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        set_seeds(42 + fold)
        tf.keras.backend.clear_session()
        
        mlp_config = CONFIGS['B'].copy()
        mlp_config.pop('encoded', None)
        mlp_config.pop('use_class_weight', None)
        mlp = build_mlp(input_dim=X_train_scaled.shape[1], **mlp_config)
        mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        
        probs = mlp.predict(X_test_scaled, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)
        
        accs.append(accuracy_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, probs))
        
    results.append({
        'Model': 'MLP (DL)',
        'Accuracy (CV)': f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
        'ROC-AUC (CV)': f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
    })
    
    # 3. MLP encoded features
    accs, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        encoder = ComplexFeatureEncoder(n_bins=8)
        encoder.fit(X_train)
        X_train_encoded = encode(X_train, encoder)
        X_test_encoded = encode(X_test, encoder)
        
        set_seeds(42 + fold)
        tf.keras.backend.clear_session()
        
        mlp_config = CONFIGS['E'].copy()
        use_cw = mlp_config.pop('use_class_weight', False)
        mlp_config.pop('encoded', False)
        
        mlp = build_mlp(input_dim=X_train_encoded.shape[1], **mlp_config)
        mlp.fit(X_train_encoded, y_train, epochs=50, batch_size=32, verbose=0)
        
        probs = mlp.predict(X_test_encoded, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)
        
        accs.append(accuracy_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, probs))
        
    results.append({
        'Model': 'MLP encoded features',
        'Accuracy (CV)': f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
        'ROC-AUC (CV)': f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
    })
    
    res_df = pd.DataFrame(results)
    res_df.to_csv('results/cv_comparison.csv', index=False)
    print(res_df)

if __name__ == '__main__':
    run_cv()
