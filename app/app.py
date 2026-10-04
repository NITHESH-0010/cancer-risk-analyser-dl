 # ============================================================
# CANCER RISK PREDICTION - FLASK BACKEND
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import joblib
import traceback

# ============================================================
# INITIALIZE FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL & SCALER
# ============================================================

import os
import joblib

try:

    # BASE DIRECTORY OF app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # MODEL PATHS
    model_path = os.path.join(
        BASE_DIR,
        "..",
        "models",
        "best_cancer_model.pkl"
    )

    scaler_path = os.path.join(
        BASE_DIR,
        "..",
        "models",
        "scaler.pkl"
    )

    # LOAD FILES
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model and Scaler Loaded Successfully")

except Exception as e:

    print("❌ Error Loading Model Files")
    print(e)

# ============================================================
# HOME ROUTE
# ============================================================

@app.route('/')
def home():

    return render_template(
        'index.html',
        prediction_text=None,
        probability_text=None
    )

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ====================================================
        # GET INPUT VALUES
        # ====================================================

        age = float(request.form['Age'])
        gender = float(request.form['Gender'])
        bmi = float(request.form['BMI'])
        smoking = float(request.form['Smoking'])
        genetic_risk = float(request.form['GeneticRisk'])
        physical_activity = float(request.form['PhysicalActivity'])
        alcohol_intake = float(request.form['AlcoholIntake'])
        cancer_history = float(request.form['CancerHistory'])

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        if age < 1 or age > 120:
            raise ValueError("Age must be between 1 and 120")

        if bmi < 10 or bmi > 70:
            raise ValueError("BMI must be between 10 and 70")

        if physical_activity < 0 or physical_activity > 40:
            raise ValueError("Physical Activity must be between 0 and 40 hrs/week")

        if alcohol_intake < 0 or alcohol_intake > 50:
            raise ValueError("Alcohol Intake must be between 0 and 50")

        # ====================================================
        # CREATE INPUT ARRAY
        # ====================================================

        input_data = np.array([[
            age,
            gender,
            bmi,
            smoking,
            genetic_risk,
            physical_activity,
            alcohol_intake,
            cancer_history
        ]])

        # ====================================================
        # SCALE INPUT DATA
        # ====================================================

        scaled_data = scaler.transform(input_data)

        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        prediction = model.predict(scaled_data)[0]

        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        probability = model.predict_proba(scaled_data)[0]

        low_risk_prob = round(probability[0] * 100, 2)
        high_risk_prob = round(probability[1] * 100, 2)

        # ====================================================
        # GENERATE RESULT
        # ====================================================

        if prediction == 1:

            prediction_text = "⚠ High Cancer Risk Detected"

            probability_text = (
                f"High Risk Probability: {high_risk_prob}%"
            )

            recommendation = (
                "Immediate medical consultation and detailed "
                "clinical screening are strongly recommended."
            )

            result_class = "high-risk"

        else:

            prediction_text = "✅ Low Cancer Risk Detected"

            probability_text = (
                f"Low Risk Probability: {low_risk_prob}%"
            )

            recommendation = (
                "Maintain healthy lifestyle habits and "
                "continue regular medical checkups."
            )

            result_class = "low-risk"

        # ====================================================
        # BMI CLASSIFICATION
        # ====================================================

        if bmi < 18.5:
            bmi_status = "Underweight"

        elif bmi < 25:
            bmi_status = "Normal Weight"

        elif bmi < 30:
            bmi_status = "Overweight"

        else:
            bmi_status = "Obese"

        # ====================================================
        # RENDER RESULT
        # ====================================================

        return render_template(

            'index.html',

            prediction_text=prediction_text,

            probability_text=probability_text,

            recommendation=recommendation,

            bmi_status=bmi_status,

            result_class=result_class,

            age=age,
            bmi=bmi,
            smoking=smoking,
            physical_activity=physical_activity,
            alcohol_intake=alcohol_intake

        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(traceback.format_exc())

        return render_template(

            'index.html',

            prediction_text="❌ Prediction Failed",

            probability_text=str(e),

            recommendation="Please verify all input fields and try again.",

            result_class="error-risk"

        )

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )