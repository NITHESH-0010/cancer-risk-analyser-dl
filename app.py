import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model_io import load_encoder, load_dl_model
from features import encode

# Must be the first Streamlit command
st.set_page_config(page_title="Cancer Risk Analyser", layout="wide")

@st.cache_resource
def get_model_and_encoder():
    encoder = load_encoder('models/feature_encoder.pkl')
    model = load_dl_model('models/cancer_dl_model_v2.keras')
    return encoder, model

@st.cache_data
def get_data_ranges():
    df = pd.read_csv("dataset/cancer_data.csv")
    ranges = {}
    for col in df.columns:
        if col != 'Diagnosis':
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'unique': df[col].dropna().unique().tolist()
            }
    return ranges

@st.cache_data
def load_metrics():
    ablation = pd.read_csv("results/ablation.csv")
    cv = pd.read_csv("results/cv_comparison.csv")
    return ablation, cv

try:
    encoder, model = get_model_and_encoder()
    ranges = get_data_ranges()
    ablation_df, cv_df = load_metrics()
except Exception as e:
    st.error(f"Failed to load model, encoder, or data: {e}")
    st.stop()

st.title("Cancer Risk Analyser")
st.markdown("**Disclaimer:** This model is trained on a likely synthetic Kaggle dataset. It has not been clinically validated and should **not** be used for medical advice or diagnostic purposes.")

tab1, tab2 = st.tabs(["Prediction", "Model Details"])

with tab1:
    st.header("Patient Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=int(ranges['Age']['min']), max_value=int(ranges['Age']['max']), value=int(ranges['Age']['mean']))
        gender = st.selectbox("Gender (0=Female, 1=Male)", options=sorted([int(x) for x in ranges['Gender']['unique']]))
        bmi = st.slider("BMI", min_value=float(ranges['BMI']['min']), max_value=float(ranges['BMI']['max']), value=float(ranges['BMI']['mean']))
        smoking = st.selectbox("Smoking (0=No, 1=Yes)", options=sorted([int(x) for x in ranges['Smoking']['unique']]))
        
    with col2:
        genetic_risk = st.selectbox("Genetic Risk (0=Low, 1=Medium, 2=High)", options=sorted([int(x) for x in ranges['GeneticRisk']['unique']]))
        physical_activity = st.slider("Physical Activity (hours/week)", min_value=float(ranges['PhysicalActivity']['min']), max_value=float(ranges['PhysicalActivity']['max']), value=float(ranges['PhysicalActivity']['mean']))
        alcohol_intake = st.slider("Alcohol Intake (units/week)", min_value=float(ranges['AlcoholIntake']['min']), max_value=float(ranges['AlcoholIntake']['max']), value=float(ranges['AlcoholIntake']['mean']))
        cancer_history = st.selectbox("Cancer History (0=No, 1=Yes)", options=sorted([int(x) for x in ranges['CancerHistory']['unique']]))
        
    if st.button("Predict Risk", type="primary"):
        # Order must match self.raw_columns exactly
        # 'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Smoking': smoking,
            'GeneticRisk': genetic_risk,
            'PhysicalActivity': physical_activity,
            'AlcoholIntake': alcohol_intake,
            'CancerHistory': cancer_history
        }])
        
        try:
            encoded_input = encode(input_data, encoder)
            prob = model.predict(encoded_input, verbose=0)[0][0]
            
            st.subheader("Prediction Results")
            st.write(f"**Predicted Probability:** {prob * 100:.1f}%")
            st.progress(float(prob))
            
            # Risk band
            if prob < 0.33:
                risk_band = "Low"
            elif prob < 0.66:
                risk_band = "Moderate"
            else:
                risk_band = "Higher"
                
            st.write(f"**Risk Band:** {risk_band}")
            st.write(f"**Predicted Class (0.5 threshold):** {1 if prob >= 0.5 else 0}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
    st.divider()
    st.subheader("Model Reliability")
    st.info("The metrics below represent the model's overall performance on the dataset, not the accuracy of this specific prediction.")
    
    # Extract encoded MLP stats
    encoded_mlp_test = ablation_df[ablation_df['Model'] == 'MLP encoded features'].iloc[0]
    encoded_mlp_cv = cv_df[cv_df['Model'] == 'MLP encoded features'].iloc[0]
    
    st.write(f"- **Test Set Accuracy:** {encoded_mlp_test['Accuracy']}")
    st.write(f"- **Test Set F1 Score:** {encoded_mlp_test['F1 Score']}")
    st.write(f"- **Test Set ROC-AUC:** {encoded_mlp_test['ROC-AUC']}")
    st.write(f"- **5-Fold CV Accuracy:** {encoded_mlp_cv['Accuracy (CV)']}")
    st.write(f"- **5-Fold CV ROC-AUC:** {encoded_mlp_cv['ROC-AUC (CV)']}")
    
with tab2:
    st.header("Model Details")
    
    st.subheader("Performance Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Test Split Results**")
        st.dataframe(ablation_df, hide_index=True)
    with c2:
        st.write("**5-Fold Cross-Validation Results**")
        st.dataframe(cv_df, hide_index=True)
        
    st.subheader("Feature Importance")
    st.write("Permutation importance computed at the raw-feature level.")
    import os
    if os.path.exists("results/permutation_importance_v2.png"):
        st.image("results/permutation_importance_v2.png")
    else:
        st.write("Feature importance plot not found.")
