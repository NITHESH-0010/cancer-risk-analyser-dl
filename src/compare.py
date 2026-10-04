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

def compare_test_split():
    # Load baselines
    try:
        baselines = pd.read_csv('results/baselines.csv')
    except:
        print("Could not find baselines.csv")
        return
        
    # Load DL results
    try:
        # Re-read to get mean +/- std
        dl_results = pd.read_csv('results/dl_results.csv', skiprows=9) 
        # Skipping the "--- Per Seed Results ---" and the seed lines.
        # Actually, it's safer to just calculate it from the file or read it properly.
    except:
        pass
    
    # Let's just read the whole file line by line to extract the summary
    dl_metrics = {}
    try:
        with open('results/dl_results.csv', 'r') as f:
            lines = f.readlines()
            summary_idx = lines.index("--- Mean and Std ---\n")
            
            # Skip the header of the summary
            for line in lines[summary_idx+2:]:
                parts = line.strip().split(',')
                metric = parts[0]
                mean_std_str = parts[3]
                # the format was: Metric,mean,std,Mean +/- Std
                dl_metrics[metric] = mean_std_str
    except Exception as e:
        print(f"Error reading dl_results.csv: {e}")
        return
    
    # We'll format the baselines similarly or just append MLP
    comparison = baselines.copy()
    comparison = comparison.astype(str) # convert all to strings for mean +/- std
    
    mlp_row = {
        'Model': 'MLP (DL)',
        'Accuracy': dl_metrics.get('Accuracy', 'N/A'),
        'Precision': dl_metrics.get('Precision', 'N/A'),
        'Recall': dl_metrics.get('Recall', 'N/A'),
        'F1 Score': dl_metrics.get('F1 Score', 'N/A'),
        'ROC-AUC': dl_metrics.get('ROC-AUC', 'N/A')
    }
    
    comparison = pd.concat([comparison, pd.DataFrame([mlp_row])], ignore_index=True)
    comparison.to_csv('results/comparison.csv', index=False)
    print("Saved results/comparison.csv")
    print(comparison)

def perform_cv():
    df = load_and_verify_data()
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis'].values
    
    binary_cols = [col for col in X.columns if X[col].nunique() <= 2]
    numeric_cols = [col for col in X.columns if X[col].nunique() > 2]
    
    # Get MLP config
    try:
        arch_df = pd.read_csv('results/arch_search.csv')
        best_config_name = arch_df.loc[arch_df['Mean Val AUC'].idxmax()]['Config']
    except:
        best_config_name = 'D'
        
    mlp_config = CONFIGS[best_config_name]
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP (DL)': None # We will build this inside the loop
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {name: {'acc': [], 'auc': []} for name in models.keys()}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"CV Fold {fold+1}/5")
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        X_train = X_train.values
        X_test = X_test.values
        
        for name, model in models.items():
            if name == 'MLP (DL)':
                set_seeds(42 + fold)
                tf.keras.backend.clear_session()
                mlp = build_mlp(**mlp_config)
                mlp.fit(
                    X_train, y_train,
                    epochs=50, # Less epochs for CV to speed it up
                    batch_size=32,
                    verbose=0
                )
                y_prob = mlp.predict(X_test, verbose=0).flatten()
                y_pred = (y_prob > 0.5).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            cv_results[name]['acc'].append(acc)
            cv_results[name]['auc'].append(auc)
            
    # Compile results
    summary = []
    
    # For plotting
    names = []
    auc_means = []
    auc_stds = []
    
    for name, metrics in cv_results.items():
        acc_mean, acc_std = np.mean(metrics['acc']), np.std(metrics['acc'])
        auc_mean, auc_std = np.mean(metrics['auc']), np.std(metrics['auc'])
        
        summary.append({
            'Model': name,
            'Accuracy (CV)': f"{acc_mean:.4f} +/- {acc_std:.4f}",
            'ROC-AUC (CV)': f"{auc_mean:.4f} +/- {auc_std:.4f}"
        })
        
        names.append(name)
        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        
    cv_df = pd.DataFrame(summary)
    cv_df.to_csv('results/cv_comparison.csv', index=False)
    print("Saved results/cv_comparison.csv")
    print(cv_df)
    
    # Plot bar chart with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(names, auc_means, yerr=auc_stds, capsize=10, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('5-Fold Cross-Validation ROC-AUC')
    plt.ylabel('ROC-AUC')
    plt.ylim([0.8, 1.0])
    plt.savefig('results/cv_auc_bar.png')
    plt.close()

if __name__ == "__main__":
    compare_test_split()
    perform_cv()
