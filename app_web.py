from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from predict import predict_proba
from shapley import compute_exact_shapley, what_if

app = Flask(__name__)

# Preload data ranges
df = pd.read_csv('dataset/cancer_data.csv')
ranges = {}
for col in df.columns:
    if col != 'Diagnosis':
        ranges[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'unique': sorted([float(x) for x in df[col].dropna().unique()])
        }

@app.route('/')
def index():
    return render_template('index.html', ranges=ranges)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    if os.path.exists('results/final_metrics.csv'):
        metrics = pd.read_csv('results/final_metrics.csv').to_dict(orient='records')
        cv = pd.read_csv('results/cv_comparison.csv').to_dict(orient='records')
        return jsonify({'test_metrics': metrics, 'cv_metrics': cv})
    return jsonify({})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model', 'Ensemble')
    
    # Validation
    row_data = {}
    for col in ranges.keys():
        if col not in data:
            return jsonify({'error': f'Missing field: {col}'}), 400
        val = float(data[col])
        if val < ranges[col]['min'] or val > ranges[col]['max']:
            return jsonify({'error': f'{col} value {val} is outside training range [{ranges[col]["min"]}, {ranges[col]["max"]}]'}), 400
        row_data[col] = val
        
    raw_df = pd.DataFrame([row_data])
    
    try:
        # Predict
        p_final = float(predict_proba(raw_df, model_name)[0])
        
        # Risk band
        if p_final < 0.33:
            band = 'Low'
            color = 'green'
        elif p_final < 0.66:
            band = 'Moderate'
            color = 'amber'
        elif p_final < 0.85:
            band = 'High'
            color = 'orange'
        else:
            band = 'Very high'
            color = 'red'
            
        # Exact Shapley
        base_value_pts, contributions_pts, final_prob_pts = compute_exact_shapley(raw_df, model_name)
        
        # What-if
        wi_results = what_if(raw_df, model_name)
        
        return jsonify({
            'probability': p_final,
            'risk_band': band,
            'risk_color': color,
            'base_value_pts': base_value_pts,
            'contributions_pts': contributions_pts,
            'final_prob_pts': final_prob_pts,
            'what_if': wi_results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
