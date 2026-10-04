import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate():
    # Load splits
    train_data = np.load('data_splits/train.npz')
    test_data = np.load('data_splits/test.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    
    # Train and evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        })
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/baselines.csv', index=False)
    
    # Print table
    print("\n--- Baseline Results ---")
    print(results_df.to_string(index=False))
    print("\nResults saved to results/baselines.csv")

if __name__ == "__main__":
    train_and_evaluate()
