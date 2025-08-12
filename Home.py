import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pipeline import preprocess, predict 
from PIL import Image

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
        background-color: #008080;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .step-card {
        background-color: #e0f7fa;
        border-left: 4px solid #008080;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>Vaccine Campaign Optimizer</h1>
    <p>Data-driven strategies for public health officials</p>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div class="card">
    <h2>About This Tool</h2>
    <p>This application helps public health officials and vaccine campaign managers optimize their outreach efforts through data analysis.</p>
</div>
""", unsafe_allow_html=True)

# Steps in cards
st.markdown("""
<div class="card">
    <h2>How It Works</h2>
    
    <div class="step-card">
        <h3>1. Upload Survey Data</h3>
        <p>Provide your CSV file containing survey responses from the target population.</p>
    </div>
    
    <div class="step-card">
        <h3>2. Analyze Predictions</h3>
        <p>Our system processes the data to predict vaccination likelihood.</p>
    </div>
    
    <div class="step-card">
        <h3>3. Explore Recommendations</h3>
        <p>Get targeted campaign strategies based on the analysis.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# File Upload Section
st.markdown("""
<div class="card">
    <h2>Upload Your Data</h2>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose survey data file (CSV format)", type=["csv"],
                           help="Upload your CSV file containing survey responses")

if uploaded is None:
    st.info("Please upload your survey data to begin analysis")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")

# Data preview
st.markdown("""
<div class="card">
    <h3>Data Preview</h3>
""", unsafe_allow_html=True)
st.dataframe(df_raw.head())

# Analysis Section
st.markdown("""
<div class="card">
    <h2>Generate Predictions</h2>
""", unsafe_allow_html=True)

if st.button("Analyze Vaccination Likelihood", type="primary", 
            use_container_width=True,
            help="Process the data and generate predictions"):
    
    with st.spinner("Analyzing data... This may take a moment"):
        # Clean and predict
        df_clean = preprocess(df_raw.copy())
        h1n1_label, seasonal_label = predict(df_clean)
        
        # Store results
        results = df_raw.copy()
        results["h1n1_label"] = h1n1_label
        results["seasonal_label"] = seasonal_label
        st.session_state["results_df"] = results
        
        # Success message
        st.balloons()
        st.success("Analysis complete! Visit the Recommendations page for campaign strategies.")
        
        # Metrics
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        st.markdown("""
        <div class="highlight-box">
            <h3>Results Summary</h3>
        """, unsafe_allow_html=True)
        
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
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close highlight-box

        # Visualization
        st.markdown("""
        <div class="card">
            <h3>Vaccination Likelihood Distribution</h3>
        """, unsafe_allow_html=True)
        
        # Matplotlib pie charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # H1N1 pie chart
        ax[0].pie([h1_vax_pct, 100-h1_vax_pct], 
                 labels=["Likely", "Unlikely"], 
                 autopct='%1.1f%%',
                 startangle=90,
                 colors=['#4dd0e1', '#80deea'],
                 explode=(0.1, 0),
                 textprops={'fontsize': 10})
        ax[0].set_title("H1N1 Vaccination", fontweight='bold')
        
        # Seasonal pie chart
        ax[1].pie([seas_vax_pct, 100-seas_vax_pct], 
                  labels=["Likely", "Unlikely"], 
                  autopct='%1.1f%%',
                  startangle=90,
                  colors=['#26c6da', '#b2ebf2'],
                  explode=(0.1, 0),
                  textprops={'fontsize': 10})
        ax[1].set_title("Seasonal Vaccination", fontweight='bold')
        
        st.pyplot(fig)
        
        # Plotly interactive chart
        fig = px.bar(
            x=["H1N1 Vaccine", "Seasonal Vaccine"],
            y=[h1_vax_pct, seas_vax_pct],
            color=["H1N1 Vaccine", "Seasonal Vaccine"],
            color_discrete_sequence=["#008080", "#4dd0e1"],
            labels={'x': 'Vaccine Type', 'y': 'Likelihood (%)'},
            text=[f"{h1_vax_pct:.1f}%", f"{seas_vax_pct:.1f}%"],
            height=400
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title="Vaccination Likelihood Comparison",
            showlegend=False,
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close card
        
        # Recommendation prompt
        st.markdown("""
        <div class="highlight-box">
            <h3>Ready for Campaign Recommendations?</h3>
            <p>Visit the Recommendations page to see targeted strategies based on this analysis.</p>
        </div>
        """, unsafe_allow_html=True)