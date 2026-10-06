import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

# Ensure src is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import encode

def get_baselines_and_quartiles(df=None, data_path='dataset/cancer_data.csv'):
    if df is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(base_dir, data_path))
    
    baseline = {}
    quartiles = {}
    
    for col in df.columns:
        if col == 'Diagnosis':
            continue
        if df[col].nunique() <= 3:
            baseline[col] = df[col].mode()[0]
        else:
            baseline[col] = df[col].median()
            
        quartiles[col] = {
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return baseline, quartiles

def get_logit(p):
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p_clipped / (1 - p_clipped)))

def get_true_logit(df, model, encoder):
    """Extracts true pre-sigmoid logit to bypass float32 saturation at 1.0"""
    encoded = encode(df, encoder)
    penultimate = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    out = penultimate.predict(encoded, verbose=0)
    W, b = model.layers[-1].get_weights()
    return float(np.dot(out, W)[0][0] + b[0])

def factor_contributions(raw_row_df, model, encoder, baseline=None, data_path='dataset/cancer_data.csv'):
    if baseline is None:
        baseline, _ = get_baselines_and_quartiles(data_path=data_path)
        
    encoded_full = encode(raw_row_df, encoder)
    p_full = float(model.predict(encoded_full, verbose=0)[0][0])
    
    # Create baseline patient
    baseline_patient = pd.DataFrame([baseline])
    # ensure correct column order
    baseline_patient = baseline_patient[raw_row_df.columns]
    
    encoded_base = encode(baseline_patient, encoder)
    p_base = float(model.predict(encoded_base, verbose=0)[0][0])
    
    # Also get logit diffs just for what_if if needed, though we can skip
    logit_full = get_true_logit(raw_row_df, model, encoder)
    
    raw_effects = {}
    for col in raw_row_df.columns:
        replaced_row = raw_row_df.copy()
        replaced_row[col] = baseline[col]
        
        encoded_replaced = encode(replaced_row, encoder)
        p_replaced = float(model.predict(encoded_replaced, verbose=0)[0][0])
        raw_effects[col] = p_full - p_replaced
        
    total_raw = sum(raw_effects.values())
    diff = p_full - p_base
    
    contributions_prob = {}
    if abs(total_raw) > 1e-6:
        for col, eff in raw_effects.items():
            contributions_prob[col] = eff * (diff / total_raw)
    else:
        for col in raw_row_df.columns:
            contributions_prob[col] = diff / len(raw_row_df.columns) if diff != 0 else 0.0
            
    # For backwards compatibility with what_if's delta_lo (not actually used by UI for chart anymore, but let's keep dict format)
    # The new UI will use contributions_prob!
    contributions_lo = {}
    for col in raw_row_df.columns:
        replaced_row = raw_row_df.copy()
        replaced_row[col] = baseline[col]
        logit_replaced = get_true_logit(replaced_row, model, encoder)
        contributions_lo[col] = logit_full - logit_replaced
        
    return p_full, contributions_lo, contributions_prob, p_base

def what_if(raw_row_df, model, encoder, quartiles=None, data_path='dataset/cancer_data.csv'):
    if quartiles is None:
        _, quartiles = get_baselines_and_quartiles(data_path=data_path)
        
    p_full = float(model.predict(encode(raw_row_df, encoder), verbose=0)[0][0])
    logit_full = get_true_logit(raw_row_df, model, encoder)
    
    results = {}
    
    def evaluate_what_if(key, row):
        p_new = float(model.predict(encode(row, encoder), verbose=0)[0][0])
        logit_new = get_true_logit(row, model, encoder)
        results[key] = {'p_new': p_new, 'delta_lo': float(logit_full - logit_new)}
        
    if raw_row_df['Smoking'].iloc[0] != 0:
        row = raw_row_df.copy()
        row['Smoking'] = 0
        evaluate_what_if('Smoking', row)
        
    lower_alc = quartiles['AlcoholIntake']['q25']
    if raw_row_df['AlcoholIntake'].iloc[0] > lower_alc:
        row = raw_row_df.copy()
        row['AlcoholIntake'] = lower_alc
        evaluate_what_if('AlcoholIntake', row)
        
    upper_pa = quartiles['PhysicalActivity']['q75']
    if raw_row_df['PhysicalActivity'].iloc[0] < upper_pa:
        row = raw_row_df.copy()
        row['PhysicalActivity'] = upper_pa
        evaluate_what_if('PhysicalActivity', row)
        
    bmi = raw_row_df['BMI'].iloc[0]
    new_bmi = None
    if bmi > 24.9:
        new_bmi = 24.9
    elif bmi < 18.5:
        new_bmi = 18.5
        
    if new_bmi is not None and quartiles['BMI']['min'] <= new_bmi <= quartiles['BMI']['max']:
        row = raw_row_df.copy()
        row['BMI'] = new_bmi
        evaluate_what_if('BMI', row)
        
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
        evaluate_what_if('Combined', combined_row)
        
    return results
