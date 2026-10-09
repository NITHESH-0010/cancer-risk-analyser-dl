from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from features import encode
from model_io import load_encoder, load_dl_model
from explain_one import factor_contributions

app = Flask(__name__)

# Global variables for model, encoder, and metrics
model = None
encoder = None
model_accuracy = 0.0
model_auc = 0.0

def load_resources():
    global model, encoder, model_accuracy, model_auc
    
    # Load model and encoder
    model_path = os.path.join(base_dir, 'models', 'cancer_dl_model_v2.keras')
    encoder_path = os.path.join(base_dir, 'models', 'feature_encoder.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = load_dl_model(model_path)
        encoder = load_encoder(encoder_path)
    else:
        print("Model or encoder not found. Please train the model first.")
        
    # Load metrics
    metrics_path = os.path.join(base_dir, 'results', 'final_metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        mlp_row = metrics_df[metrics_df['Model'] == 'Encoded MLP']
        if not mlp_row.empty:
            model_accuracy = mlp_row.iloc[0]['Accuracy'] * 100
            model_auc = mlp_row.iloc[0]['ROC-AUC']
            
load_resources()

PRECAUTIONS = {
    'Age': 'Regular screenings and check-ups are recommended as risk naturally increases with age.',
    'BMI': 'Maintain a balanced diet and consult a nutritionist to achieve a healthy weight.',
    'Smoking': 'Consider enrolling in a smoking cessation program to reduce risk significantly.',
    'GeneticRisk': 'Discuss regular specialized screenings or genetic counseling with your doctor.',
    'PhysicalActivity': 'Aim for at least 150 minutes of moderate aerobic activity every week.',
    'AlcoholIntake': 'Limit alcohol consumption following national health guidelines.',
    'CancerHistory': 'Ensure rigorous follow-up appointments and monitoring with your oncologist.'
}

REASONS = {
    'Age': 'Advanced age is a natural risk factor.',
    'BMI': 'Higher BMI can contribute to metabolic stress.',
    'Smoking': 'Smoking introduces carcinogens into the body.',
    'GeneticRisk': 'Inherited mutations can increase susceptibility.',
    'PhysicalActivity': 'Low activity levels affect overall immunity and health.',
    'AlcoholIntake': 'High alcohol intake is linked to cell damage.',
    'CancerHistory': 'Previous history increases the likelihood of recurrence.'
}

def get_risk_band(prob):
    if prob < 33:
        return 'Low'
    elif prob < 66:
        return 'Moderate'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Accept Gender but ignore it
        _ = data.get('Gender', 'Other')
        
        # Extract features
        features = ['Age', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
        row = {}
        for f in features:
            if f not in data or data[f] is None:
                return jsonify({'error': f'Missing value for {f}'}), 400
            
            try:
                val = float(data[f])
            except ValueError:
                return jsonify({'error': f'Invalid value for {f}'}), 400
                
            row[f] = val
            
        # Basic validation
        if not (0 <= row['Age'] <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if not (10 <= row['BMI'] <= 60):
            return jsonify({'error': 'BMI must be between 10 and 60'}), 400
        if row['PhysicalActivity'] < 0:
            return jsonify({'error': 'Physical Activity cannot be negative'}), 400
        if row['AlcoholIntake'] < 0:
            return jsonify({'error': 'Alcohol Intake cannot be negative'}), 400
            
        df_row = pd.DataFrame([row])
        
        # Explain and predict
        p_full, _, contributions_prob, _ = factor_contributions(df_row, model, encoder)
        
        probability = p_full * 100
        
        # Sort contributions by absolute impact
        contribs_list = []
        for f, impact in contributions_prob.items():
            pts = impact * 100
            contribs_list.append({
                'name': f,
                'points': round(pts, 1),
                'abs_points': abs(pts),
                'reason': REASONS.get(f, 'Affects cancer risk.'),
                'precaution': PRECAUTIONS.get(f, 'Consult your doctor for advice.')
            })
            
        contribs_list.sort(key=lambda x: x['abs_points'], reverse=True)
        
        # Format for output
        for c in contribs_list:
            del c['abs_points']
            
        return jsonify({
            'probability': round(probability, 1),
            'risk_band': get_risk_band(probability),
            'predicted_class': 1 if probability >= 50 else 0,
            'model_accuracy': round(model_accuracy, 1),
            'model_auc': round(model_auc, 3),
            'contributions': contribs_list
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
