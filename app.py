import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model_io import load_encoder, load_dl_model
from features import encode
from explain_one import factor_contributions, what_if

st.set_page_config(page_title="Cancer Risk Analyser", layout="wide", page_icon="🩺")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #172a45 100%);
        color: #e6f1ff;
    }
    
    .css-1d391kg, .css-18e3th9 { /* Sidebar / Tab headers */
        background: transparent;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        background: #112240;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    h1, h2, h3, h4, h5, h6, .stMarkdown p {
        color: #ccd6f6;
    }
    
    .stButton>button {
        background-color: #64ffda;
        color: #0a192f;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #52e0c4;
        color: #0a192f;
    }
    
    .risk-card {
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #0a192f;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
    .risk-low { background: linear-gradient(135deg, #a8e063 0%, #56ab2f 100%); }
    .risk-mod { background: linear-gradient(135deg, #ffd194 0%, #70e1f5 100%); /* Adjusted to more amber */ }
    .risk-mod { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
    .risk-high { background: linear-gradient(135deg, #ff4b2b 0%, #ff416c 100%); }
    
    .risk-score {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #020c1b;
        color: #8892b0;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

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

# Header
st.markdown("<h1>🩺 Cancer Risk Analyser DL</h1>", unsafe_allow_html=True)
st.markdown("<h5>An interactive deep learning model for evaluating lifestyle and health factors.</h5>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predict", "Model details"])

with tab1:
    
    st.markdown("### Patient Data Input")
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
        
    if st.button("Predict Risk"):
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
        
        # 1. Explanation Logic
        p_full, contributions = factor_contributions(input_data, model, encoder)
        what_if_res = what_if(input_data, model, encoder)
        
        st.markdown("---")
        
        # 2. Result Section
        if p_full < 0.33:
            risk_class = "risk-low"
            risk_text = "Low"
        elif p_full < 0.66:
            risk_class = "risk-mod"
            risk_text = "Moderate"
        else:
            risk_class = "risk-high"
            risk_text = "Higher"
            
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h2>Predicted Risk Band: {risk_text}</h2>
            <div class="risk-score">{p_full * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(p_full))
        
        st.markdown(f"**Predicted Class (0.5 threshold):** {1 if p_full >= 0.5 else 0}")
        st.caption("Note: This score is a model output reflecting patterns in the data, not a calibrated real-world probability (training used class weights to balance outcomes).")
        
        st.markdown("---")
        st.markdown("### Factors behind this score")
        
        # Plot contributions
        c_series = pd.Series(contributions).sort_values()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_alpha(0.0) # Transparent bg
        ax.set_facecolor("none")
        
        colors = ['#ff416c' if x > 0 else '#56ab2f' for x in c_series.values]
        c_series.plot(kind='barh', color=colors, ax=ax)
        ax.set_xlabel('Contribution to Score (probability delta)')
        ax.tick_params(axis='both', colors='#ccd6f6')
        ax.xaxis.label.set_color('#ccd6f6')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#ccd6f6')
            
        st.pyplot(fig)
        
        # 2-3 sentences naming the top drivers
        sorted_factors = c_series.sort_values(ascending=False)
        top_raisers = sorted_factors[sorted_factors > 0]
        
        if len(top_raisers) > 0:
            top_factor = top_raisers.index[0]
            sentence1 = f"**{top_factor}** is the biggest factor raising this score."
        else:
            sentence1 = "No factors are significantly raising the score above baseline."
            
        st.markdown(f"{sentence1} The model finds patterns in the data and does not prove cause and effect.")
        
        st.markdown("---")
        st.markdown("### Things you can change")
        
        st.markdown("**Modifiable Factors & Potential Impact**")
        
        def display_what_if(factor, text, key):
            if key in what_if_res:
                new_p = what_if_res[key]
                if new_p < p_full - 0.001:
                    st.success(f"- **{factor}:** {text} Estimated new score: {new_p*100:.1f}%.")
                else:
                    st.info(f"- **{factor}:** {text} Changing this factor does not show an improvement in the model's score for your specific profile.")
        
        display_what_if("Smoking", "Stopping smoking generally supports better overall health.", "Smoking")
        display_what_if("Alcohol Intake", "Moderating alcohol intake can contribute to long-term wellness.", "AlcoholIntake")
        display_what_if("Physical Activity", "Regular physical activity is beneficial for a healthy lifestyle.", "PhysicalActivity")
        display_what_if("BMI", "Maintaining a healthy weight can positively impact well-being.", "BMI")
        
        if 'Combined' in what_if_res:
            st.markdown(f"**Combined Impact:** Adopting all applicable lifestyle changes above could shift the score to **{what_if_res['Combined']*100:.1f}%**.")
            
        st.markdown("**Not changeable:**")
        st.markdown("Age, Gender, Genetic Risk, and Cancer History are fixed factors.")
        
with tab2:
    st.header("Model Details")
    
    st.subheader("Model Reliability")
    st.info("The metrics below represent the model's overall performance on the dataset, not the accuracy of a single prediction.")
    
    encoded_mlp_test = ablation_df[ablation_df['Model'] == 'MLP encoded features'].iloc[0]
    encoded_mlp_cv = cv_df[cv_df['Model'] == 'MLP encoded features'].iloc[0]
    
    st.write(f"- **Test Set Accuracy:** {encoded_mlp_test['Accuracy']}")
    st.write(f"- **Test Set F1 Score:** {encoded_mlp_test['F1 Score']}")
    st.write(f"- **Test Set ROC-AUC:** {encoded_mlp_test['ROC-AUC']}")
    st.write(f"- **5-Fold CV Accuracy:** {encoded_mlp_cv['Accuracy (CV)']}")
    st.write(f"- **5-Fold CV ROC-AUC:** {encoded_mlp_cv['ROC-AUC (CV)']}")
    
    st.markdown("---")
    
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
    if os.path.exists("results/permutation_importance_v2.png"):
        st.image("results/permutation_importance_v2.png")
    else:
        st.write("Feature importance plot not found.")

st.markdown("""
<div class="footer">
    Trained on a probably synthetic Kaggle dataset; not clinically validated and not medical advice. Scores are model outputs, not diagnoses.
</div>
""", unsafe_allow_html=True)
