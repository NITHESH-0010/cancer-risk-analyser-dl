import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import encode

def get_baselines_and_quartiles(df=None, data_path='dataset/cancer_data.csv'):
    if df is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(base_dir, data_path))
    
    baseline = {}
    quartiles = {}
    
    # Categorical mode, Numeric median
    for col in df.columns:
        if col == 'Diagnosis':
            continue
        if df[col].nunique() <= 3: # Categorical
            baseline[col] = df[col].mode()[0]
        else: # Numeric
            baseline[col] = df[col].median()
            
        quartiles[col] = {
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return baseline, quartiles

def factor_contributions(raw_row_df, model, encoder, baseline=None, data_path='dataset/cancer_data.csv'):
    if baseline is None:
        baseline, _ = get_baselines_and_quartiles(data_path=data_path)
        
    encoded_full = encode(raw_row_df, encoder)
    p_full = model.predict(encoded_full, verbose=0)[0][0]
    
    contributions = {}
    for col in raw_row_df.columns:
        replaced_row = raw_row_df.copy()
        replaced_row[col] = baseline[col]
        encoded_replaced = encode(replaced_row, encoder)
        p_replaced = model.predict(encoded_replaced, verbose=0)[0][0]
        delta = p_full - p_replaced
        contributions[col] = float(delta)
        
    return float(p_full), contributions

def what_if(raw_row_df, model, encoder, quartiles=None, data_path='dataset/cancer_data.csv'):
    if quartiles is None:
        _, quartiles = get_baselines_and_quartiles(data_path=data_path)
        
    results = {}
    
    # Smoking -> 0
    if raw_row_df['Smoking'].iloc[0] != 0:
        row = raw_row_df.copy()
        row['Smoking'] = 0
        p_smoke = model.predict(encode(row, encoder), verbose=0)[0][0]
        results['Smoking'] = float(p_smoke)
        
    # AlcoholIntake -> lower quartile
    lower_alc = quartiles['AlcoholIntake']['q25']
    if raw_row_df['AlcoholIntake'].iloc[0] > lower_alc:
        row = raw_row_df.copy()
        row['AlcoholIntake'] = lower_alc
        p_alc = model.predict(encode(row, encoder), verbose=0)[0][0]
        results['AlcoholIntake'] = float(p_alc)
        
    # PhysicalActivity -> upper quartile
    upper_pa = quartiles['PhysicalActivity']['q75']
    if raw_row_df['PhysicalActivity'].iloc[0] < upper_pa:
        row = raw_row_df.copy()
        row['PhysicalActivity'] = upper_pa
        p_pa = model.predict(encode(row, encoder), verbose=0)[0][0]
        results['PhysicalActivity'] = float(p_pa)
        
    # BMI -> clamp into 18.5 - 24.9
    bmi = raw_row_df['BMI'].iloc[0]
    new_bmi = None
    if bmi > 24.9:
        new_bmi = 24.9
    elif bmi < 18.5:
        new_bmi = 18.5
        
    if new_bmi is not None and quartiles['BMI']['min'] <= new_bmi <= quartiles['BMI']['max']:
        row = raw_row_df.copy()
        row['BMI'] = new_bmi
        p_bmi = model.predict(encode(row, encoder), verbose=0)[0][0]
        results['BMI'] = float(p_bmi)
        
    # All changes combined
    combined_row = raw_row_df.copy()
    changed = False
    if raw_row_df['Smoking'].iloc[0] != 0:
        combined_row['Smoking'] = 0
        changed = True
    if raw_row_df['AlcoholIntake'].iloc[0] > lower_alc:
        combined_row['AlcoholIntake'] = lower_alc
        changed = True
    if raw_row_df['PhysicalActivity'].iloc[0] < upper_pa:
        combined_row['PhysicalActivity'] = upper_pa
        changed = True
    if new_bmi is not None and quartiles['BMI']['min'] <= new_bmi <= quartiles['BMI']['max']:
        combined_row['BMI'] = new_bmi
        changed = True
        
    if changed:
        p_comb = model.predict(encode(combined_row, encoder), verbose=0)[0][0]
        results['Combined'] = float(p_comb)
        
    return results
