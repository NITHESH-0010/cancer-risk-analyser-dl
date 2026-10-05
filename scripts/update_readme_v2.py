import pandas as pd
import re

def update_readme():
    with open('README.md', 'r') as f:
        content = f.read()
        
    # Load ablation.csv for Test Split
    ablation = pd.read_csv('results/ablation.csv')
    test_rows = []
    # Print header
    test_rows.append("| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |")
    test_rows.append("| --- | --- | --- | --- | --- | --- |")
    
    # We want these models in this order if possible
    order_test = ['MLP raw features', 'MLP encoded features', 'Logistic Regression', 'Random Forest', 'Gradient Boosting']
    
    for model_name in order_test:
        row = ablation[ablation['Model'] == model_name]
        if not row.empty:
            r = row.iloc[0]
            # format floats if they aren't strings
            def fmt(x):
                try:
                    return f"{float(x):.4f}"
                except:
                    return str(x)
            test_rows.append(f"| {r['Model']} | {fmt(r['Accuracy'])} | {fmt(r['Precision'])} | {fmt(r['Recall'])} | {fmt(r['F1 Score'])} | {fmt(r['ROC-AUC'])} |")
            
    test_table = "\n".join(test_rows)
    
    # Load cv_comparison.csv
    cv_comp = pd.read_csv('results/cv_comparison.csv')
    cv_rows = []
    cv_rows.append("| Model | Accuracy (CV) | ROC-AUC (CV) |")
    cv_rows.append("| --- | --- | --- |")
    order_cv = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'MLP (DL)', 'MLP encoded features']
    for model_name in order_cv:
        row = cv_comp[cv_comp['Model'] == model_name]
        if not row.empty:
            r = row.iloc[0]
            cv_rows.append(f"| {r['Model']} | {r['Accuracy (CV)']} | {r['ROC-AUC (CV)']} |")
            
    cv_table = "\n".join(cv_rows)
    
    # Replace Test Split Results
    content = re.sub(r'### Test Split Results\n\| Model \| Accuracy.*?(?=\n### 5-Fold Cross-Validation Results)', 
                     f'### Test Split Results\n{test_table}\n', content, flags=re.DOTALL)
                     
    # Replace CV Results
    content = re.sub(r'### 5-Fold Cross-Validation Results\n\| Model \| Accuracy \(CV\).*?(?=\n### Conclusion)', 
                     f'### 5-Fold Cross-Validation Results\n{cv_table}\n', content, flags=re.DOTALL)
                     
    # Remove "vastly bridged the gap" or similar sentences if they exist (already checked, but just in case)
    content = content.replace("vastly bridged the gap", "improved performance")
    
    with open('README.md', 'w') as f:
        f.write(content)
        
if __name__ == "__main__":
    update_readme()
