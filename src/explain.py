import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import joblib
from features import ComplexFeatureEncoder, QuantilePiecewiseLinearEncoder

def get_mlp_scorer(model, encoder):
    def scorer(estimator, X, y):
        X_encoded = encoder.transform(X)
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
    
    feature_names = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
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

if __name__ == "__main__":
    explain_mlp()
