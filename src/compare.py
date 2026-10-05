import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from data import load_and_verify_data
from model import build_mlp
from search import CONFIGS, set_seeds
from features import ComplexFeatureEncoder, encode

def make_ablation():
    # Load baselines
    try:
        baselines = pd.read_csv('results/baselines.csv').astype(str)
    except:
        return
        
    # Read old DL results
    try:
        with open('results/dl_results_v1.csv', 'r') as f:
            pass # Actually we only have results/dl_results.csv from v1
    except:
        pass
        
    mlp_v1 = {'Model': 'MLP raw features', 'Accuracy': '0.8569 +/- 0.0119', 'Precision': '0.8097 +/- 0.0098', 'Recall': '0.8000 +/- 0.0326', 'F1 Score': '0.8046 +/- 0.0194', 'ROC-AUC': '0.9257 +/- 0.0036'}

    # Extract new DL results from dl_results.csv
    dl_metrics = {}
    try:
        with open('results/dl_results.csv', 'r') as f:
            lines = f.readlines()
            summary_idx = lines.index("--- Mean and Std ---\n")
            for line in lines[summary_idx+2:]:
                if line.strip():
                    parts = line.strip().split(',')
                    dl_metrics[parts[0]] = parts[3]
    except Exception as e:
        print(f"Error reading dl_results.csv: {e}")
        
    mlp_v2 = {
        'Model': 'MLP encoded features',
        'Accuracy': dl_metrics.get('Accuracy', 'N/A'),
        'Precision': dl_metrics.get('Precision', 'N/A'),
        'Recall': dl_metrics.get('Recall', 'N/A'),
        'F1 Score': dl_metrics.get('F1 Score', 'N/A'),
        'ROC-AUC': dl_metrics.get('ROC-AUC', 'N/A')
    }
    
    ablation = pd.concat([pd.DataFrame([mlp_v1, mlp_v2]), baselines], ignore_index=True)
    ablation.to_csv('results/ablation.csv', index=False)
    print("Saved results/ablation.csv")
    print(ablation)

def perform_cv_encoded():
    df = load_and_verify_data()
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis'].values
    
    # Get MLP config
    try:
        arch_df = pd.read_csv('results/arch_search_v2.csv')
        encoded_df = arch_df[arch_df['Encoded'] == True]
        best_config_name = encoded_df.loc[encoded_df['Mean Val AUC'].idxmax()]['Config']
    except:
        best_config_name = 'E'
        
    mlp_config = CONFIGS[best_config_name]
    is_encoded = mlp_config.pop('encoded')
    use_cw = mlp_config.pop('use_class_weight')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"CV Encoded Fold {fold+1}/5")
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit encoder INSIDE each fold to avoid leakage
        encoder = ComplexFeatureEncoder(n_bins=8)
        encoder.fit(X_train)
        X_train_encoded = encode(X_train, encoder)
        X_test_encoded = encode(X_test, encoder)
        
        set_seeds(42 + fold)
        tf.keras.backend.clear_session()
        
        input_dim = X_train_encoded.shape[1]
        mlp = build_mlp(input_dim=input_dim, **mlp_config)
        
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        cw = {cls: weight for cls, weight in zip(classes, weights)} if use_cw else None
        
        mlp.fit(
            X_train_encoded, y_train,
            epochs=50,
            batch_size=32,
            class_weight=cw,
            verbose=0
        )
        y_prob = mlp.predict(X_test_encoded, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    
    # Load old cv_comparison.csv
    cv_df = pd.read_csv('results/cv_comparison.csv')
    
    new_row = pd.DataFrame([{
        'Model': 'MLP encoded features',
        'Accuracy (CV)': f"{acc_mean:.4f} +/- {acc_std:.4f}",
        'ROC-AUC (CV)': f"{auc_mean:.4f} +/- {auc_std:.4f}"
    }])
    
    cv_df = pd.concat([cv_df, new_row], ignore_index=True)
    cv_df.to_csv('results/cv_comparison.csv', index=False)
    print(cv_df)
    
    # Re-plot
    names = cv_df['Model'].tolist()
    auc_means = [float(x.split(' +/- ')[0]) for x in cv_df['ROC-AUC (CV)']]
    auc_stds = [float(x.split(' +/- ')[1]) for x in cv_df['ROC-AUC (CV)']]
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, auc_means, yerr=auc_stds, capsize=10, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
    plt.title('5-Fold Cross-Validation ROC-AUC')
    plt.ylabel('ROC-AUC')
    plt.ylim([0.8, 1.0])
    plt.savefig('results/cv_auc_bar.png')
    plt.close()

if __name__ == "__main__":
    make_ablation()
    perform_cv_encoded()
