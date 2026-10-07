import pandas as pd
import os

def load_and_verify_data(filepath="dataset/cancer_data.csv"):
    # Load dataset with relative path
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    
    # Check missing values and duplicates
    missing_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    print("\n--- Missing Values ---")
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values.")
    
    print(f"\n--- Duplicates ---")
    print(f"Number of duplicate rows: {duplicate_count}")
    
    # Assert every feature column has more than one unique value
    print("\n--- Checking for constant columns ---")
    for col in df.columns:
        unique_vals = df[col].nunique()
        assert unique_vals > 1, f"Assertion failed: Column '{col}' is constant (only {unique_vals} unique value)."
    print("All columns have more than one unique value.")
    
    print("\n--- Class Balance (Diagnosis) ---")
    if 'Diagnosis' in df.columns:
        print(df['Diagnosis'].value_counts())
    else:
        print("Diagnosis column not found.")
    
    print("\n--- CancerHistory Value Counts ---")
    if 'CancerHistory' in df.columns:
        print(df['CancerHistory'].value_counts())
    else:
        print("CancerHistory column not found.")
        
    return df

if __name__ == "__main__":
    load_and_verify_data()
