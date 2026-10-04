import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from model import build_mlp

# Candidate configs
CONFIGS = {
    'A': {'hidden_units': [32], 'dropout': 0.2, 'l2': 0.0, 'lr': 1e-3},
    'B': {'hidden_units': [64, 32], 'dropout': 0.3, 'l2': 0.0, 'lr': 1e-3},
    'C': {'hidden_units': [128, 64, 32], 'dropout': 0.3, 'l2': 0.0, 'lr': 1e-3},
    'D': {'hidden_units': [64, 32], 'dropout': 0.3, 'l2': 1e-3, 'lr': 1e-3}
}

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def perform_search():
    # Load splits
    train_data = np.load('data_splits/train.npz')
    val_data = np.load('data_splits/val.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    
    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"Computed class weights: {class_weight}")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0
    )
    
    results = []
    
    seeds = [42, 123, 999]
    
    for config_name, config in CONFIGS.items():
        print(f"\n--- Testing Config {config_name}: {config} ---")
        auc_scores = []
        recall_scores = []
        
        for seed in seeds:
            set_seeds(seed)
            
            model = build_mlp(**config)
            
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate on validation set
            y_val_prob = model.predict(X_val, verbose=0).flatten()
            y_val_pred = (y_val_prob > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_val_prob)
            rec = recall_score(y_val, y_val_pred)
            
            auc_scores.append(auc)
            recall_scores.append(rec)
            print(f"Seed {seed}: val_auc={auc:.4f}, val_recall={rec:.4f}")
            
        mean_auc = np.mean(auc_scores)
        mean_recall = np.mean(recall_scores)
        
        print(f"Config {config_name} Mean val_auc={mean_auc:.4f}, Mean val_recall={mean_recall:.4f}")
        
        results.append({
            'Config': config_name,
            'Hidden Units': str(config['hidden_units']),
            'Dropout': config['dropout'],
            'L2': config['l2'],
            'Mean Val AUC': mean_auc,
            'Mean Val Recall': mean_recall
        })
        
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/arch_search.csv', index=False)
    
    # Pick best config
    best_config_name = results_df.loc[results_df['Mean Val AUC'].idxmax()]['Config']
    print(f"\nBest config by Mean Val AUC is: {best_config_name}")
    print(results_df.to_string(index=False))
    
    return best_config_name

if __name__ == "__main__":
    perform_search()
