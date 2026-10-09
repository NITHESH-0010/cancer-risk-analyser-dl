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
background_df = None
bg_prob = 0.0
penultimate_model = None

# Using metrics from the TEST set, evaluated separately
model_accuracy = 81.3
model_auc = 0.905
model_brier = 0.1176

def load_resources():
    global model, encoder, calibrator, background_df, bg_prob, penultimate_model
    model_path = os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras')
    encoder_path = os.path.join(base_dir, 'models', 'feature_encoder.pkl')
    calibrator_path = os.path.join(base_dir, 'models', 'platt_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = load_dl_model(model_path)
        encoder = load_encoder(encoder_path)
    if os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
        
    train_data = np.load(os.path.join(base_dir, 'data_splits', 'train_raw.npz'), allow_pickle=True)
    X_train = pd.DataFrame(train_data['X'], columns=['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
    
    np.random.seed(42)
    idx = np.random.choice(len(X_train), 15, replace=False)
    background_df = X_train.iloc[idx].reset_index(drop=True)
    
    import tensorflow as tf
    penultimate_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    logits = get_logits(background_df)
    if calibrator:
        probs = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    else:
        probs = 1 / (1 + np.exp(-logits))
    bg_prob = np.mean(probs)

def get_logits(df_batch):
    encoded = encode(df_batch, encoder)
    out = penultimate_model.predict(encoded, verbose=0)
    W, b = model.layers[-1].get_weights()
    logits = np.dot(out, W) + b
    return logits.flatten()

load_resources()

def compute_shapley_logits(df_row):
    features = ['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
    
    np.random.seed(123)
    permutations = [np.random.permutation(features) for _ in range(5)]
    
    batch_rows = []
    user_row = df_row.iloc[0].to_dict()
    
    for i in range(len(background_df)):
        bg_row = background_df.iloc[i].to_dict()
        
        for perm in permutations:
            current_row = bg_row.copy()
            batch_rows.append(current_row.copy())
            
            for f in perm:
                current_row[f] = user_row[f]
                batch_rows.append(current_row.copy())

    df_batch = pd.DataFrame(batch_rows)
    logits = get_logits(df_batch)
    
    attributions = {f: 0.0 for f in features}
    idx = 0
    
    for i in range(len(background_df)):
        for perm in permutations:
            prev_logit = logits[idx]
            idx += 1
            for f in perm:
                curr_logit = logits[idx]
                idx += 1
                attributions[f] += (curr_logit - prev_logit)
                prev_logit = curr_logit
                
    total_runs = len(background_df) * len(permutations)
    for f in features:
        attributions[f] /= total_runs
        
    return attributions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        _ = data.get('Gender', 'Other')
        
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
        
        # User logit
        user_logit = get_logits(df_row)[0]
        if calibrator:
            user_prob = calibrator.predict_proba([[user_logit]])[0][1]
        else:
            user_prob = 1 / (1 + np.exp(-user_logit))
            
        display_p = np.clip(user_prob, 0.01, 0.99)
        probability_pct = display_p * 100
        
        # Base probability to distribute diff
        base_display_p = np.clip(bg_prob, 0.01, 0.99)
        prob_diff_pct = probability_pct - (base_display_p * 100)
        
        # Shapley on logits
        shap_logits = compute_shapley_logits(df_row)
        total_shap = sum(shap_logits.values())
        
        contribs_list = []
        for f in features:
            if abs(total_shap) > 1e-6:
                share = shap_logits[f] / total_shap
            else:
                share = 1.0 / len(features)
                
            # Convert share to percentage points
            diff_pts = share * prob_diff_pct
            
            # Guidelines Check
            is_healthy = False
            if f == 'BMI' and (18.5 <= row[f] <= 24.9): is_healthy = True
            elif f == 'PhysicalActivity' and row[f] >= 2.5: is_healthy = True
            elif f == 'AlcoholIntake' and row[f] == 0: is_healthy = True
            elif f == 'Smoking' and row[f] == 0: is_healthy = True
            
            # Text Explanations
            if diff_pts > 0:
                if is_healthy:
                    reason = "Small effect in the model for this value."
                    precaution = "Keep up the great work in this area!"
                    print(f"Flagging inconsistency: {f} value {row[f]} is healthy but model shows +{diff_pts:.1f} pts")
                else:
                    if f == 'Age':
                        reason = "Age cannot be changed."
                        precaution = "Keep up regular check-ups."
                    elif f == 'BMI':
                        reason = f"Your BMI ({row[f]}) raises risk."
                        precaution = "Consult a nutritionist or doctor to manage weight."
                    elif f == 'Smoking':
                        reason = "Smoking is a major carcinogen."
                        precaution = "Enroll in a smoking cessation program."
                    elif f == 'GeneticRisk':
                        reason = "Your genetic profile elevates your risk."
                        precaution = "Discuss specialized screening with your doctor."
                    elif f == 'PhysicalActivity':
                        reason = f"Your activity level ({row[f]} hrs/wk) raises risk."
                        precaution = "Aim to increase weekly physical activity."
                    elif f == 'AlcoholIntake':
                        reason = f"Your alcohol intake ({row[f]} units) raises risk."
                        precaution = "Limit alcohol consumption to lower your risk."
                    else: # CancerHistory
                        reason = "A previous history of cancer raises recurrence risk."
                        precaution = "Maintain rigorous follow-up appointments."
            else:
                if f == 'Age':
                    reason = "Age cannot be changed."
                    precaution = "Keep up regular check-ups."
                else:
                    reason = f"Your {f} value lowers or maintains your risk."
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
            'probability': round(probability_pct, 1),
            'risk_band': get_risk_band(probability_pct),
            'predicted_class': 1 if probability_pct >= 50 else 0,
            'model_accuracy': round(model_accuracy, 1),
            'model_auc': round(model_auc, 3),
            'contributions': contribs_list
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
