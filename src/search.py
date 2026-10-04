import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import joblib

from model import build_mlp
from features import ComplexFeatureEncoder, QuantilePiecewiseLinearEncoder

# Candidate configs
CONFIGS = {
    'A': {'hidden_units': [32], 'dropout': 0.2, 'l2': 0.0, 'lr': 1e-3, 'encoded': False, 'use_class_weight': True},
    'B': {'hidden_units': [64, 32], 'dropout': 0.3, 'l2': 0.0, 'lr': 1e-3, 'encoded': False, 'use_class_weight': True},
    'C': {'hidden_units': [128, 64, 32], 'dropout': 0.3, 'l2': 0.0, 'lr': 1e-3, 'encoded': False, 'use_class_weight': True},
    'D': {'hidden_units': [64, 32], 'dropout': 0.3, 'l2': 1e-3, 'lr': 1e-3, 'encoded': False, 'use_class_weight': True},
    # Encoded feature configs
    'E': {'hidden_units': [32], 'dropout': 0.1, 'l2': 0.0, 'lr': 1e-3, 'encoded': True, 'use_class_weight': True},
    'F': {'hidden_units': [64, 32], 'dropout': 0.2, 'l2': 0.0, 'lr': 1e-3, 'encoded': True, 'use_class_weight': True},
    'G': {'hidden_units': [64, 32], 'dropout': 0.1, 'l2': 1e-3, 'lr': 1e-3, 'encoded': True, 'use_class_weight': True},
    'H': {'hidden_units': [64, 32], 'dropout': 0.1, 'l2': 1e-3, 'lr': 1e-3, 'encoded': True, 'use_class_weight': False}
}

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def perform_search():
    # Load raw splits
    train_raw = np.load('data_splits/train_raw.npz')
    val_raw = np.load('data_splits/val_raw.npz')
    X_train_raw, y_train = train_raw['X'], train_raw['y']
    X_val_raw, y_val = val_raw['X'], val_raw['y']
    
    # Load scaled splits (from v1)
    train_scaled = np.load('data_splits/train.npz')
    val_scaled = np.load('data_splits/val.npz')
    X_train_scaled = train_scaled['X']
    X_val_scaled = val_scaled['X']
    
    # Load encoder
    encoder = joblib.load('models/feature_encoder.pkl')
    X_train_encoded = encoder.transform(X_train_raw)
    X_val_encoded = encoder.transform(X_val_raw)
    
    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0
    )
    
    results = []
    seeds = [42, 123, 999]
    
    for config_name, config_params in CONFIGS.items():
        print(f"\n--- Testing Config {config_name} ---")
        auc_scores = []
        recall_scores = []
        
        is_encoded = config_params.pop('encoded')
        use_cw = config_params.pop('use_class_weight')
        
        cw = class_weight_dict if use_cw else None
        
        if is_encoded:
            X_tr, X_v = X_train_encoded, X_val_encoded
        else:
            X_tr, X_v = X_train_scaled, X_val_scaled
            
        input_dim = X_tr.shape[1]
        
        for seed in seeds:
            set_seeds(seed)
            model = build_mlp(input_dim=input_dim, **config_params)
            
            model.fit(
                X_tr, y_train,
                validation_data=(X_v, y_val),
                epochs=100,
                batch_size=32,
                class_weight=cw,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate on validation set
            y_val_prob = model.predict(X_v, verbose=0).flatten()
            y_val_pred = (y_val_prob > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_val_prob)
            rec = recall_score(y_val, y_val_pred)
            
            auc_scores.append(auc)
            recall_scores.append(rec)
            
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        mean_recall = np.mean(recall_scores)
        
        print(f"Config {config_name} Mean val_auc={mean_auc:.4f} +/- {std_auc:.4f}, Mean val_recall={mean_recall:.4f}")
        
        results.append({
            'Config': config_name,
            'Encoded': is_encoded,
            'Use_CW': use_cw,
            'Hidden Units': str(config_params['hidden_units']),
            'Dropout': config_params['dropout'],
            'L2': config_params['l2'],
            'Mean Val AUC': mean_auc,
            'Std Val AUC': std_auc,
            'Mean Val Recall': mean_recall
        })
        
        # Restore params
        config_params['encoded'] = is_encoded
        config_params['use_class_weight'] = use_cw
        
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/arch_search_v2.csv', index=False)
    
    # Pick best config considering standard deviation
    max_auc = results_df['Mean Val AUC'].max()
    best_idx = results_df['Mean Val AUC'].idxmax()
    best_std = results_df.loc[best_idx]['Std Val AUC']
    
    close_configs = results_df[results_df['Mean Val AUC'] >= (max_auc - best_std)]
    if len(close_configs) > 1:
        print(f"\nMultiple configs are within one standard deviation ({best_std:.4f}) of the max AUC ({max_auc:.4f}):")
        print(close_configs['Config'].tolist())
        print("We cannot confidently claim one is the absolute best. Picking the one with highest mean just as a default.")
    else:
        print(f"\nBest config is clearly {results_df.loc[best_idx]['Config']}.")
        
    best_config_name = results_df.loc[best_idx]['Config']
    print(f"\nBest config by pure Mean Val AUC is: {best_config_name}")
    print(results_df.to_string(index=False))
    
    return best_config_name

if __name__ == "__main__":
    perform_search()
