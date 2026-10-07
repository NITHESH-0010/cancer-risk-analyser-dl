import pandas as pd
import numpy as np
import sys
import os

from src.model_io import load_encoder, load_dl_model
from src.explain_one import factor_contributions, what_if
from src.features import encode

def format_prob(p):
    if p < 0.01: return "<1%"
    elif p > 0.99: return ">99%"
    else: return f"{p*100:.1f}%"

encoder = load_encoder('models/feature_encoder.pkl')
model = load_dl_model('models/cancer_dl_model_v2.keras')

data_A = pd.DataFrame([{
    'Age': 25, 'BMI': 22, 'Smoking': 0, 'GeneticRisk': 0, 
    'PhysicalActivity': 8, 'AlcoholIntake': 0.5, 'CancerHistory': 0
}])

data_B = pd.DataFrame([{
    'Age': 65, 'BMI': 35, 'Smoking': 1, 'GeneticRisk': 2, 
    'PhysicalActivity': 1, 'AlcoholIntake': 4, 'CancerHistory': 1
}])

def print_test(name, df):
    print(f"\n=== Test {name} ===")
    p_full, contrib_lo, contrib_prob = factor_contributions(df, model, encoder)
    print(f"Probability: {format_prob(p_full)} (raw: {p_full:.6f})")
    
    sorted_contribs = sorted(contrib_lo.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 3 Log-Odds Contributions:")
    for k, v in sorted_contribs[:3]:
        print(f"  {k}: {v:+.4f}")
        
    res = what_if(df, model, encoder)
    print("What-if Results:")
    for k, v in res.items():
        print(f"  {k}: new prob = {format_prob(v['p_new'])}, log-odds drop = {v['delta_lo']:.4f}")

print_test('A', data_A)
print_test('B', data_B)

# Calibration Check
print("\n=== Calibration Check ===")
test_raw = np.load('data_splits/test_raw.npz')
X_test_raw, y_test = test_raw['X'], test_raw['y']

X_test_encoded = encode(X_test_raw, encoder)
preds = model.predict(X_test_encoded, verbose=0).flatten()

fraction_below_01 = np.mean(preds < 0.01)
fraction_above_99 = np.mean(preds > 0.99)
print(f"Fraction predicted < 0.01: {fraction_below_01:.4f}")
print(f"Fraction predicted > 0.99: {fraction_above_99:.4f}")

mean_pred_class_0 = np.mean(preds[y_test == 0])
mean_pred_class_1 = np.mean(preds[y_test == 1])
print(f"Mean predicted prob for true class 0: {mean_pred_class_0:.4f}")
print(f"Mean predicted prob for true class 1: {mean_pred_class_1:.4f}")
