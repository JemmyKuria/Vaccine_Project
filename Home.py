import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pipeline import preprocess, predict 

# Page Configuration
st.set_page_config(
    page_title="Vaccine Campaign Optimizer", 
    page_icon="💉",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        color: #2e7d32;
        text-align: center;
        margin-bottom: 30px;
    }
    .section {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        width: 100%;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">Vaccine Campaign Optimizer</h1>', unsafe_allow_html=True)

# About Us Section
st.markdown("""
<div class="section">
    <h2>About This Tool</h2>
    <p>This application helps <strong>public health officials</strong> and <strong>vaccine campaign managers</strong> optimize their outreach efforts by:</p>
    <ol>
        <li>Identifying populations with low vaccination likelihood</li>
        <li>Analyzing key behavioral and demographic factors affecting vaccine uptake</li>
        <li>Recommending targeted messaging strategies for different audience segments</li>
    </ol>
    
    <h3>How It Works</h3>
    <p>1. <strong>Upload</strong> your survey data (CSV format)</p>
    <p>2. <strong>Analyze</strong> the predictions to understand vaccination likelihood</p>
    <p>3. <strong>Explore</strong> the recommendations page for targeted campaign strategies</p>
    
    <p><em>Designed for Ministries of Health and public health organizations to maximize vaccine campaign effectiveness.</em></p>
</div>
""", unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="section"><h2>1. Upload Survey Data</h2></div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Choose your survey data file (CSV format)", type=["csv"])
if uploaded is None:
    st.info("Please upload your survey data to begin analysis")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")

# Show raw data preview
st.markdown("### Data Preview")
st.dataframe(df_raw.head())

# Processing Section
st.markdown('<div class="section"><h2>2. Generate Predictions</h2></div>', unsafe_allow_html=True)
if st.button("Analyze Vaccination Likelihood", type="primary"):
    with st.spinner("Processing data and generating insights..."):
        # Clean and predict
        df_clean = preprocess(df_raw.copy())
        h1n1_label, seasonal_label = predict(df_clean)
        
        # Create final results
        results = df_raw.copy()
        results["h1n1_label"] = h1n1_label
        results["seasonal_label"] = seasonal_label
        
        # Store in session state
        st.session_state["results_df"] = results
        
        # Success message
        st.balloons()
        st.success("Analysis complete! Visit the Recommendations page for campaign strategies.")
        
        # Metrics Display
        st.markdown("### Vaccination Likelihood Results")
        
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        # Metrics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Respondents</h3>
                <h1>{total:,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>H1N1 Vaccine Likely</h3>
                <h1>{h1_vax_pct:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Seasonal Vaccine Likely</h3>
                <h1>{seas_vax_pct:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### Vaccination Distribution")
        fig = px.pie(
            names=["H1N1 Likely", "H1N1 Unlikely", "Seasonal Likely", "Seasonal Unlikely"],
            values=[h1_vax_pct, 100-h1_vax_pct, seas_vax_pct, 100-seas_vax_pct],
            color=["H1N1 Likely", "H1N1 Unlikely", "Seasonal Likely", "Seasonal Unlikely"],
            color_discrete_map={
                "H1N1 Likely": "#81c784",
                "H1N1 Unlikely": "#ff8a65",
                "Seasonal Likely": "#66bb6a",
                "Seasonal Unlikely": "#ff7043"
            },
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)