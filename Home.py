import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline import preprocess, predict

# =========================
# Page Config
# =========================
st.set_page_config(page_title="🏠 Home", layout="wide")

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
/* Page background */
.main {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', sans-serif;
}

/* Header Section */
.header-container {
    background: linear-gradient(135deg, #2c3e50, #3498db);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1.5rem;
}

/* Buttons */
.stButton > button {
    background-color: #3498db;
    color: white;
    font-size: 16px;
    padding: 8px 24px;
    border-radius: 8px;
    transition: all 0.3s;
    border: none;
}
.stButton > button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
}

/* File uploader */
.stFileUploader {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #ccc;
    text-align: center;
}

/* KPI Metric Cards */
.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    text-align: center;
}

/* About the model section */
.model-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Feature cards inside About section */
.feature-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class='header-container'>
    <h1>Public Health Vaccine Predictor</h1>
    <p>Predicting H1N1 & Seasonal Flu Vaccination Likelihood</p>
</div>
""", unsafe_allow_html=True)

# =========================
# About the Model Section
# =========================
st.markdown("""
<div class='model-section'>
    <h2 style='color: #2c3e50;'>About the Model</h2>
    <p style='font-size: 16px;'>
    Our predictive model analyzes survey responses to estimate an individual's likelihood of receiving 
    H1N1 and seasonal flu vaccinations. This tool helps public health officials identify populations 
    that may need additional education or outreach to improve vaccination rates.
    </p>
    
    <h3 style='color: #2c3e50; margin-top: 20px;'>Key Features</h3>
    
    <div class='feature-card'>
        <h4 style='color: #3498db;'>📊 Predictive Accuracy</h4>
        <p>Our model achieves 85% accuracy in predicting vaccination likelihood based on behavioral and demographic factors.</p>
    </div>
    
    <div class='feature-card'>
        <h4 style='color: #3498db;'>🔍 Important Factors</h4>
        <ul>
            <li>Perceived vaccine effectiveness</li>
            <li>Previous vaccination history</li>
            <li>Doctor recommendations</li>
            <li>Demographic information</li>
            <li>Health insurance status</li>
        </ul>
    </div>
    
    <div class='feature-card'>
        <h4 style='color: #3498db;'>🛠️ Technical Details</h4>
        <p>
        The model uses a Gradient Boosting Classifier trained on CDC survey data. 
        It was validated using 10-fold cross-validation with an AUC score of 0.87.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Upload Data Section
# =========================
st.markdown("### 📂 Upload Your Survey Data")
with st.expander("Upload Data", expanded=True):
    uploaded = st.file_uploader("Upload survey CSV file", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write("Raw preview (first 5 rows):")
        st.dataframe(df_raw.head())
    else:
        st.info("Please upload a CSV file to get started")

# =========================
# Prediction Section
# =========================
st.markdown("### 🚀 Generate Predictions")
with st.expander("Process Data and Predict", expanded=True):
    if uploaded and st.button("Process Data and Predict"):
        with st.spinner('Processing data and generating predictions...'):
            df_clean = preprocess(df_raw.copy())
            h1n1_label, seasonal_label = predict(df_clean)

            results = df_raw.copy()
            results["h1n1_label"] = h1n1_label
            results["seasonal_label"] = seasonal_label
            st.session_state["results_df"] = results

            total = len(results)
            h1_vax_pct = results["h1n1_label"].mean() * 100
            seas_vax_pct = results["seasonal_label"].mean() * 100

            # KPI Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><h3>{total:,}</h3><p>Total Respondents</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><h3>{h1_vax_pct:.1f}%</h3><p>H1N1 Likely</p></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><h3>{seas_vax_pct:.1f}%</h3><p>Seasonal Likely</p></div>", unsafe_allow_html=True)

            # Donut Chart
            chart_df = pd.DataFrame({
                "Category": ["H1N1 Likely", "H1N1 Unlikely", "Seasonal Likely", "Seasonal Unlikely"],
                "Percentage": [h1_vax_pct, 100 - h1_vax_pct, seas_vax_pct, 100 - seas_vax_pct]
            })
            fig = px.pie(chart_df, values="Percentage", names="Category", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
