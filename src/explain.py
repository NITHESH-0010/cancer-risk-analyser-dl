import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import joblib
from features import ComplexFeatureEncoder, QuantilePiecewiseLinearEncoder, encode

def get_mlp_scorer(model, encoder):
    def scorer(estimator, X, y):
        X_encoded = encode(X, encoder)
        y_prob = model.predict(X_encoded, verbose=0).flatten()
        return roc_auc_score(y, y_prob)
    return scorer

class DummyEstimator:
    def fit(self, X, y):
        pass

def explain_mlp():
    # Load model v2
    model = load_model('models/cancer_dl_model_v2.keras')
    
    # Load encoder
    encoder = joblib.load('models/feature_encoder.pkl')
    
    # Load raw validation data
    val_raw = np.load('data_splits/val_raw.npz')
    X_val_raw, y_val = val_raw['X'], val_raw['y']
    
    feature_names = ['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
    # The columns passed to encoder transform must match these feature names in order, but it was passed as dataframe in data.py
    # So we should convert X_val_raw to DataFrame
    X_val_df = pd.DataFrame(X_val_raw, columns=feature_names)
    
    dummy = DummyEstimator()
    scorer = get_mlp_scorer(model, encoder)
    
    print("Computing permutation importance on validation set (grouped back to original features)...")
    result = permutation_importance(
        dummy, X_val_df, y_val, scoring=scorer, n_repeats=10, random_state=42, n_jobs=1
    )
    
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    ax.set_title("Permutation Importances (Encoded MLP on Validation Set)")
    ax.set_xlabel("Decrease in AUC score")
    fig.tight_layout()
    plt.savefig('results/permutation_importance_v2.png')
    plt.close()
    
    print("\nFeature Importances (Mean +/- Std):")
    for i in sorted_idx[::-1]:
        print(f"{feature_names[i]}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")
        
    print("\nSaved results/permutation_importance_v2.png")
    
    # Drop-column check
    print("\n--- Running Drop-Column Check for Age and GeneticRisk ---")
    train_raw = np.load('data_splits/train_raw.npz')
    X_train_raw, y_train = train_raw['X'], train_raw['y']
    X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    
    # Drop Age and GeneticRisk
    cols_to_drop = ['Age', 'GeneticRisk']
    X_train_dropped = X_train_df.drop(columns=cols_to_drop)
    X_val_dropped = X_val_df.drop(columns=cols_to_drop)
    
    # Simple encoder for the dropped data (just standard scale for simplicity, or we can use the same logic minus those)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_dropped)
    X_val_s = scaler.transform(X_val_dropped)
    
    from model import build_mlp
    import os
    os.environ['PYTHONHASHSEED'] = '42'
    import random
    random.seed(42)
    tf.random.set_seed(42)
    
    mlp = build_mlp(input_dim=X_train_s.shape[1], hidden_units=[64, 32], dropout=0.1, l2=1e-3, lr=1e-3)
    mlp.fit(X_train_s, y_train, epochs=50, batch_size=32, verbose=0)
    
    y_prob_dropped = mlp.predict(X_val_s, verbose=0).flatten()
    auc_dropped = roc_auc_score(y_val, y_prob_dropped)
    
    print(f"Validation AUC after dropping {cols_to_drop}: {auc_dropped:.4f}")
    
    # Evaluate full model for reference
    y_prob_full = model.predict(encode(X_val_raw, encoder), verbose=0).flatten()
    auc_full = roc_auc_score(y_val, y_prob_full)
    print(f"Validation AUC of full encoded model: {auc_full:.4f}")
    print(f"Drop in AUC: {auc_full - auc_dropped:.4f}")

if __name__ == "__main__":
    explain_mlp()
