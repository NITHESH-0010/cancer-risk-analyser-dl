import os
import sys
import numpy as np
import pandas as pd
import itertools
from functools import lru_cache

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from predict import predict_proba

def get_background_data(seed=42):
    train_raw = np.load(os.path.join(base_dir, 'data_splits', 'train_raw.npz'))
    X_train = train_raw['X']
    cols = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
    df = pd.DataFrame(X_train, columns=cols)
    return df.sample(n=50, random_state=seed)

def compute_exact_shapley(raw_row, model_name='Ensemble', background_df=None):
    if background_df is None:
        background_df = get_background_data()
        
    features = list(raw_row.columns)
    N = len(features)
    assert N == 8, "Expected exactly 8 features"
    
    # Precompute subset weights
    import math
    def weight(S_len):
        return math.factorial(S_len) * math.factorial(N - S_len - 1) / math.factorial(N)
        
    # Generate all 256 subsets
    all_subsets = []
    for r in range(N + 1):
        for subset in itertools.combinations(features, r):
            all_subsets.append(set(subset))
            
    # For each subset, create 50 rows
    # Total 256 * 50 = 12800 rows
    
    # To optimize, let's construct the batch DataFrame
    # For each feature j, it takes raw_row's value if j in subset, else background's value
    num_bg = len(background_df)
    total_rows = len(all_subsets) * num_bg
    
    batch_data = {}
    for f in features:
        batch_data[f] = np.empty(total_rows)
        
    idx = 0
    for subset in all_subsets:
        for i in range(num_bg):
            for f in features:
                if f in subset:
                    batch_data[f][idx] = raw_row[f].iloc[0]
                else:
                    batch_data[f][idx] = background_df[f].iloc[i]
            idx += 1
            
    batch_df = pd.DataFrame(batch_data)
    
    # Predict all at once
    preds = predict_proba(batch_df, model_name)
    
    # v(S) is the average prediction over the 50 background samples
    v = {}
    idx = 0
    for subset in all_subsets:
        v[frozenset(subset)] = np.mean(preds[idx : idx + num_bg])
        idx += num_bg
        
    base_value = v[frozenset()]
    final_prob = v[frozenset(features)]
    
    # Compute Shapley values
    contributions = {f: 0.0 for f in features}
    
    for S in all_subsets:
        S_fs = frozenset(S)
        S_len = len(S)
        
        for f in features:
            if f not in S_fs:
                w = weight(S_len)
                S_union_f = S_fs.union([f])
                contributions[f] += w * (v[S_union_f] - v[S_fs])
                
    # Convert to percentage points
    base_value_pts = base_value * 100
    final_prob_pts = final_prob * 100
    contributions_pts = {k: val * 100 for k, val in contributions.items()}
    
    # Check additivity
    sum_contrib = sum(contributions_pts.values())
    assert abs(base_value_pts + sum_contrib - final_prob_pts) < 1e-4, "Additivity check failed"
    
    return base_value_pts, contributions_pts, final_prob_pts

def what_if(raw_row, model_name='Ensemble'):
    # For modifiable features only
    features = ['Smoking', 'AlcoholIntake', 'PhysicalActivity', 'BMI']
    
    background = get_background_data()
    q25_alc = background['AlcoholIntake'].quantile(0.25)
    q75_pa = background['PhysicalActivity'].quantile(0.75)
    
    orig_prob = predict_proba(raw_row, model_name)[0]
    results = {}
    
    # Helper to evaluate
    def eval_scenario(name, row):
        new_p = predict_proba(row, model_name)[0]
        pts_change = (new_p - orig_prob) * 100
        results[name] = {'new_prob': new_p * 100, 'pts_change': pts_change}
        
    # Smoking
    if raw_row['Smoking'].iloc[0] != 0:
        row = raw_row.copy()
        row['Smoking'] = 0
        eval_scenario('Smoking', row)
        
    # Alcohol
    if raw_row['AlcoholIntake'].iloc[0] > q25_alc:
        row = raw_row.copy()
        row['AlcoholIntake'] = q25_alc
        eval_scenario('AlcoholIntake', row)
        
    # PA
    if raw_row['PhysicalActivity'].iloc[0] < q75_pa:
        row = raw_row.copy()
        row['PhysicalActivity'] = q75_pa
        eval_scenario('PhysicalActivity', row)
        
    # BMI
    bmi = raw_row['BMI'].iloc[0]
    new_bmi = None
    if bmi > 24.9: new_bmi = 24.9
    elif bmi < 18.5: new_bmi = 18.5
    
    if new_bmi is not None and background['BMI'].min() <= new_bmi <= background['BMI'].max():
        row = raw_row.copy()
        row['BMI'] = new_bmi
        eval_scenario('BMI', row)
        
    # Combined
    changed = False
    comb_row = raw_row.copy()
    if raw_row['Smoking'].iloc[0] != 0:
        comb_row['Smoking'] = 0
        changed = True
    if raw_row['AlcoholIntake'].iloc[0] > q25_alc:
        comb_row['AlcoholIntake'] = q25_alc
        changed = True
    if raw_row['PhysicalActivity'].iloc[0] < q75_pa:
        comb_row['PhysicalActivity'] = q75_pa
        changed = True
    if new_bmi is not None and background['BMI'].min() <= new_bmi <= background['BMI'].max():
        comb_row['BMI'] = new_bmi
        changed = True
        
    if changed:
        eval_scenario('Combined', comb_row)
        
    return results
