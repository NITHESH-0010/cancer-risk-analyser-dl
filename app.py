import json
import joblib
import pandas as pd
import numpy as np
import os
import sys
from flask import Flask, render_template, request, jsonify

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from features import encode
from model_io import load_encoder, load_dl_model

app = Flask(__name__)

model = None
encoder = None
calibrator = None
model_accuracy = 83.6  # Fixed from the calibration script output
model_auc = 0.899

def load_resources():
    global model, encoder, calibrator
    model_path = os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras')
    encoder_path = os.path.join(base_dir, 'models', 'feature_encoder.pkl')
    calibrator_path = os.path.join(base_dir, 'models', 'platt_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = load_dl_model(model_path)
        encoder = load_encoder(encoder_path)
    if os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)

load_resources()

# Reference healthy values
REFERENCE = {
    'Age': 51.0,               # Median
    'BMI': 27.6,               # Median
    'PhysicalActivity': 4.8,   # Median
    'AlcoholIntake': 2.4,      # Median
    'Smoking': 0,              # Lowest-risk
    'GeneticRisk': 0,          # Lowest-risk
    'CancerHistory': 0         # Lowest-risk
}

from explain_one import get_true_logit

def predict_prob(df_row):
    encoded = encode(df_row, encoder)
    p = model.predict(encoded, verbose=0).flatten()[0]
    if calibrator:
        logit = get_true_logit(df_row, model, encoder)
        p = calibrator.predict_proba([[logit]])[0][1]
    return p

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        _ = data.get('Gender', 'Other') # Accept Gender, completely ignored
        
        features = ['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
        row = {}
        for f in features:
            if f not in data or data[f] is None:
                return jsonify({'error': f'Missing value for {f}'}), 400
            try:
                row[f] = float(data[f])
            except ValueError:
                return jsonify({'error': f'Invalid value for {f}'}), 400
                
        df_row = pd.DataFrame([row])
        p_full = predict_prob(df_row)
        
        # Cap the final displayed probability
        display_p = np.clip(p_full, 0.01, 0.99)
        probability = display_p * 100
        
        contribs_list = []
        
        for f in features:
            # Replace only THIS feature with the reference healthy value
            df_ref = df_row.copy()
            df_ref.at[0, f] = REFERENCE[f]
            
            p_ref = predict_prob(df_ref)
            
            # The difference is how much THIS feature changed the probability vs its healthy baseline
            # Positive diff means the user's value RAISES risk vs healthy.
            diff_pts = (p_full - p_ref) * 100
            
            # Dynamic text based on sign
            if diff_pts > 0.5:
                # Raises risk
                if f == 'Age':
                    reason = f"Your age ({row[f]}) increases risk compared to the median."
                    precaution = "Regular screenings are highly recommended."
                elif f == 'BMI':
                    reason = f"Your BMI ({row[f]}) is a risk factor."
                    precaution = "Consult a nutritionist or doctor to manage weight."
                elif f == 'Smoking':
                    reason = "Smoking is a major carcinogen."
                    precaution = "Enroll in a smoking cessation program."
                elif f == 'GeneticRisk':
                    reason = "Your genetic profile elevates your risk."
                    precaution = "Discuss specialized screening with your doctor."
                elif f == 'PhysicalActivity':
                    reason = f"Your activity level ({row[f]} hrs/wk) is lower than average."
                    precaution = "Aim to increase weekly physical activity."
                elif f == 'AlcoholIntake':
                    reason = f"Your alcohol intake ({row[f]} units) is higher than recommended."
                    precaution = "Limit alcohol consumption to lower your risk."
                else: # CancerHistory
                    reason = "A previous history of cancer raises recurrence risk."
                    precaution = "Maintain rigorous follow-up appointments."
            else:
                # Neutral or lowers risk
                reason = f"Your {f} is at or better than the healthy reference."
                precaution = "Keep up the great work in this area!"
                
            contribs_list.append({
                'name': f,
                'points': round(diff_pts, 1),
                'abs_points': abs(diff_pts),
                'reason': reason,
                'precaution': precaution
            })
            
        contribs_list.sort(key=lambda x: x['abs_points'], reverse=True)
        for c in contribs_list:
            del c['abs_points']
            
        def get_risk_band(prob):
            if prob < 33: return 'Low'
            elif prob < 66: return 'Moderate'
            else: return 'High'
            
        return jsonify({
            'probability': round(probability, 1),
            'risk_band': get_risk_band(probability),
            'predicted_class': 1 if probability >= 50 else 0,
            'model_accuracy': model_accuracy,
            'model_auc': model_auc,
            'contributions': contribs_list
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
