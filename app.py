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
    
    .block-container {
        padding-bottom: 90px !important;
    }

    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
        color: #E6EEF8 !important;
        font-weight: 500;
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
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.2rem;
        background-color: rgba(255, 255, 255, 0.2);
        margin-bottom: 10px;
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

import plotly.graph_objects as go
import plotly.express as px

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
        # Update app to use probability diffs for the chart
        p_full, contributions_prob, p_base = factor_contributions(input_data, model, encoder)
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
            <div class="risk-badge">Risk Band: {risk_text}</div>
            <div class="risk-score">Predicted risk: {p_full * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(p_full))
        
        st.caption(f"Note: This score is a model output reflecting patterns in the data, not a calibrated real-world probability. (Baseline risk: {p_base * 100:.1f}%)")
        
        st.markdown("---")
        st.markdown("### Factors behind this score")
        st.caption("How each factor shifts the predicted risk relative to the average baseline profile.")
        
        # Plot contributions using Plotly
        # Sort by absolute impact
        items = list(contributions_prob.items())
        items.sort(key=lambda x: abs(x[1]), reverse=False) # Ascending absolute value for horizontal bar chart
        
        features = [x[0] for x in items]
        values = [x[1] * 100 for x in items]
        colors = ['#ff416c' if v > 0 else '#56ab2f' for v in values]
        texts = [f"+{v:.1f} pts" if v > 0 else f"{v:.1f} pts" for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=texts,
            textposition='auto',
        ))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccd6f6'),
            xaxis_title="Percentage Points Impact on Risk",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2-3 sentences naming the top drivers and prevention tips
        # Sort by actual risk-raising impact
        sorted_raisers = sorted([i for i in items if i[1] > 0.01], key=lambda x: x[1], reverse=True)
        
        if len(sorted_raisers) > 0:
            st.markdown("#### Top Risk-Raising Factors")
            
            tips = {
                'Smoking': ("Smoking damages DNA and limits the body's ability to heal.", "Quitting smoking is the single best action to lower risk."),
                'CancerHistory': ("Previous cancer treatments or genetics increase future risk.", "Keep up with regular screenings and medical check-ups."),
                'GeneticRisk': ("Certain gene variants predispose cells to mutations.", "Discuss more frequent screenings with your doctor."),
                'Gender': ("The dataset patterns show a correlation (likely synthetic).", "Maintain overall health (this is likely a dataset artifact)."),
                'Age': ("Cellular damage accumulates over time.", "Focus on healthy aging with diet and exercise."),
                'BMI': ("Higher BMI is linked to inflammation.", "Aim for a balanced diet and active lifestyle."),
                'AlcoholIntake': ("Alcohol breaks down into harmful chemicals.", "Limit alcohol intake to moderate levels or avoid entirely."),
                'PhysicalActivity': ("Low activity can affect hormones and immune function.", "Incorporate at least 150 minutes of moderate activity weekly.")
            }
            
            for i in range(min(3, len(sorted_raisers))):
                factor = sorted_raisers[i][0]
                cause, tip = tips.get(factor, ("Contributes to risk patterns.", "Maintain healthy lifestyle habits."))
                st.markdown(f"**{factor}:** {cause} **Tip:** {tip}")
        else:
            st.markdown("No significant risk-raising factors detected compared to the baseline.")
        
        st.markdown("---")
        st.markdown("### Things you can change")
        
        st.markdown("**Modifiable Factors & Potential Impact**")
        
        def display_what_if(factor, text, key):
            if key in what_if_res:
                res = what_if_res[key]
                new_p = res['p_new']
                delta_lo = res['delta_lo']
                
                if delta_lo > 0.1:
                    if new_p > 0.99:
                        st.success(f"- **{factor}:** {text} The score stays very high even with this change, but the risk score still drops by {delta_lo:.2f} on the log-odds scale.")
                    else:
                        st.success(f"- **{factor}:** {text} Estimated new score: {format_prob(new_p)} (risk score drops by {delta_lo:.2f}).")
                else:
                    st.info(f"- **{factor}:** {text} Changing this factor shows no improvement in the model's score for your specific profile.")
        
        display_what_if("Smoking", "Stopping smoking generally supports better overall health.", "Smoking")
        display_what_if("Alcohol Intake", "Moderating alcohol intake can contribute to long-term wellness.", "AlcoholIntake")
        display_what_if("Physical Activity", "Regular physical activity is beneficial for a healthy lifestyle.", "PhysicalActivity")
        display_what_if("BMI", "Maintaining a healthy weight can positively impact well-being.", "BMI")
        
        if 'Combined' in what_if_res:
            res_c = what_if_res['Combined']
            new_p_c = res_c['p_new']
            st.markdown(f"**Combined Impact:** Adopting all applicable lifestyle changes above could shift the score to **{format_prob(new_p_c)}**.")
            
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
