import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

def get_mlp_scorer(model):
    def scorer(estimator, X, y):
        # estimator is dummy here, we use the captured model
        y_prob = model.predict(X, verbose=0).flatten()
        return roc_auc_score(y, y_prob)
    return scorer

class DummyEstimator:
    def fit(self, X, y):
        pass

def explain_mlp():
    # Load model
    model = load_model('models/cancer_dl_model.keras')
    
    # Load validation data
    val_data = np.load('data_splits/val.npz')
    X_val, y_val = val_data['X'], val_data['y']
    
    # Feature names
    import joblib
    try:
        feature_names = joblib.load('data_splits/feature_names.pkl')
    except:
        feature_names = [f'Feature {i}' for i in range(X_val.shape[1])]
    
    # Dummy estimator wrapper for permutation importance
    dummy = DummyEstimator()
    scorer = get_mlp_scorer(model)
    
    # Compute permutation importance
    print("Computing permutation importance on validation set...")
    result = permutation_importance(
        dummy, X_val, y_val, scoring=scorer, n_repeats=10, random_state=42, n_jobs=1
    )
    
    # Sort indices
    sorted_idx = result.importances_mean.argsort()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    ax.set_title("Permutation Importances (MLP on Validation Set)")
    ax.set_xlabel("Decrease in AUC score")
    fig.tight_layout()
    plt.savefig('results/permutation_importance.png')
    plt.close()
    
    # Print out
    print("\nFeature Importances (Mean +/- Std):")
    for i in sorted_idx[::-1]:
        print(f"{feature_names[i]}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")
        
    print("\nSaved results/permutation_importance.png")

if __name__ == "__main__":
    explain_mlp()
