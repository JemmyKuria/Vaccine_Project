import streamlit as st
import pandas as pd
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
    .header {
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .step-number {
        background-color: #2e7d32;
        color: white;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="header">Vaccine Campaign Optimizer</h1>', unsafe_allow_html=True)

# About This Tool Section
with st.container():
    st.markdown("""
    <div class="section">
        <h2>About This Tool</h2>
        <p>This application helps <strong>public health officials</strong> and <strong>vaccine campaign managers</strong> optimize their outreach efforts by:</p>
        <ol style="padding-left: 1.2rem;">
            <li>Identifying populations with low vaccination likelihood</li>
            <li>Analyzing key behavioral and demographic factors affecting vaccine uptake</li>
            <li>Recommending targeted messaging strategies for different audience segments</li>
        </ol>
        
        <h3>How It Works</h3>
        <p><span class="step-number">1</span> <strong>Upload</strong> your survey data (CSV format)</p>
        <p><span class="step-number">2</span> <strong>Analyze</strong> the predictions to understand vaccination likelihood</p>
        <p><span class="step-number">3</span> <strong>Explore</strong> the recommendations page for targeted campaign strategies</p>
        
        <p style="margin-top: 1.5rem;"><em>Designed for Ministries of Health and public health organizations to maximize vaccine campaign effectiveness.</em></p>
    </div>
    """, unsafe_allow_html=True)

# File Upload Section
with st.container():
    st.markdown('<div class="section"><h2>1. Upload Survey Data</h2></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose your survey data file (CSV format)", type=["csv"], label_visibility="collapsed")
    
    if uploaded is None:
        st.info("Please upload your survey data to begin analysis")
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")
    
    # Data preview
    st.markdown("#### Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

# Analysis Section
with st.container():
    st.markdown('<div class="section"><h2>2. Generate Predictions</h2></div>', unsafe_allow_html=True)
    
    if st.button("Analyze Vaccination Likelihood", type="primary", use_container_width=True):
        with st.spinner("Processing data and generating insights..."):
            # Clean and predict
            df_clean = preprocess(df_raw.copy())
            h1n1_label, seasonal_label = predict(df_clean)
            
            # Store results
            results = df_raw.copy()
            results["h1n1_label"] = h1n1_label
            results["seasonal_label"] = seasonal_label
            st.session_state["results_df"] = results
            
            # Display results
            st.balloons()
            st.success("Analysis complete! Visit the Recommendations page for campaign strategies.")
            
            # Metrics
            total = len(results)
            h1_vax_pct = results["h1n1_label"].mean() * 100
            seas_vax_pct = results["seasonal_label"].mean() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Respondents</h4>
                    <h2>{total:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>H1N1 Vaccine Likely</h4>
                    <h2>{h1_vax_pct:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Seasonal Vaccine Likely</h4>
                    <h2>{seas_vax_pct:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.markdown("#### Vaccination Distribution")
            fig = px.pie(
                names=["H1N1 Likely", "H1N1 Unlikely", "Seasonal Likely", "Seasonal Unlikely"],
                values=[h1_vax_pct, 100-h1_vax_pct, seas_vax_pct, 100-seas_vax_pct],
                color_discrete_sequence=["#81c784", "#ff8a65", "#66bb6a", "#ff7043"],
                hole=0.4,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)






      # Display summary metrics with charts
    st.subheader("Vaccination Likelihood Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # H1N1 vaccination likelihood
    ax[0].pie([h1_vax_pct, 100 - h1_vax_pct], 
                labels=["Likely", "Unlikely"], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['#4CAF50', '#f44336'],
                explode=(0.1, 0))
    ax[0].set_title("H1N1 Vaccination Likelihood")

    # Seasonal vaccination likelihood
    ax[1].pie([seas_vax_pct, 100 - seas_vax_pct], 
                labels=["Likely", "Unlikely"], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['#4CAF50', '#f44336'],
                explode=(0.1, 0))
    ax[1].set_title("Seasonal Vaccination Likelihood")

    st.pyplot(fig)