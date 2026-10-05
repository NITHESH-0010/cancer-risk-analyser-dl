import pandas as pd
import re

def df_to_markdown(df):
    md = f"| {' | '.join(df.columns)} |\n"
    md += f"| {' | '.join(['---'] * len(df.columns))} |\n"
    for _, row in df.iterrows():
        md += f"| {' | '.join([str(x) for x in row.values])} |\n"
    return md

def update_readme_v2():
    try:
        ablation_df = pd.read_csv('results/ablation.csv')
        cv_df = pd.read_csv('results/cv_comparison.csv')
        cv_df = cv_df.drop_duplicates(subset=['Model'])
        cv_df.to_csv('results/cv_comparison.csv', index=False)
        
        ablation_md = df_to_markdown(ablation_df)
        cv_md = df_to_markdown(cv_df)
        
        new_section = "## Why the first MLP lagged\n"
        new_section += "A deep dive into the features revealed that the diagnosis rates change in sharp steps rather than smoothly (e.g., around Age 50, BMI 25). "
        new_section += "These sharp steps suggest synthetic, rule-based data generation.\n\n"
        new_section += "Because standard Neural Networks struggle to learn sharp step functions from raw numeric features, "
        new_section += "we introduced a `QuantilePiecewiseLinearEncoder`. "
        new_section += "This explicitly models these step-like nonlinearities by mapping the numeric features into bins.\n\n"
        new_section += "Here are the reconciled results for the new Encoded MLP on both the Test split and 5-Fold Cross-Validation, presented side by side with classical models:\n\n"
        new_section += "### Test Split Results\n"
        new_section += ablation_md + "\n"
        new_section += "### 5-Fold Cross-Validation Results\n"
        new_section += cv_md + "\n"
        new_section += "### Conclusion (Updated)\n"
        new_section += "The engineered features improved the MLP's performance over raw features. "
        new_section += "However, Gradient Boosting and Random Forest remain very competitive (and often superior) on this tabular dataset, reinforcing that tree ensembles are a highly robust default for such data.\n"

        with open('README.md', 'r') as f:
            content = f.read()

        # Find where "## Why the first MLP lagged" starts
        idx = content.find("## Why the first MLP lagged")
        if idx != -1:
            content = content[:idx] + new_section
        else:
            content += "\n" + new_section
            
        with open('README.md', 'w') as f:
            f.write(content)
            
        print("Updated README.md with v2 text correctly")
    except Exception as e:
        print(f"Error updating README: {e}")

if __name__ == "__main__":
    update_readme_v2()
