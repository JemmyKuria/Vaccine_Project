import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import preprocess, predict

# Page Configuration
st.set_page_config(
    page_title="Vaccine Campaign Optimizer", 
    page_icon="💉",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
    /* General Page Styling */
    .main {
        background-color: #f4f9f9;
    }
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
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .steps-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .steps-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1rem;
    }
    .step-card {
        background-color: #e0f7fa;
        border-left: 4px solid #008080;
        border-radius: 8px;
        padding: 1.5rem;
        flex: 1;
        min-width: 0;
    }
    .step-number {
        background-color: #008080;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-weight: bold;
    }
    @media (max-width: 768px) {
        .steps-container {
            flex-direction: column;
        }
    }

    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    /* Teal Button Styling */
    .stButton > button {
        background-color: #008080 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #006666 !important;
        color: white !important;
    }
    
    @media (max-width: 768px) {
        .steps-container {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="header">
    <h1>Vaccine Campaign Optimizer</h1>
    <p>Data-driven strategies for public health officials</p>
</div>
""", unsafe_allow_html=True)

# =========================
# About Section
# =========================
st.markdown("""
<div class="card">
    <h2>About This Tool</h2>
    <p>This application helps public health officials and vaccine campaign managers optimize their outreach efforts through data analysis.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# How It Works Section
# =========================
st.markdown("""
<div class="card">
    <h2>How It Works</h2>
</div>
""", unsafe_allow_html=True)

# Create columns for the step cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">1</span>
            <h3 style="margin: 0;">Upload</h3>
        </div>
        <p style="margin: 0;">Provide your CSV file containing survey responses</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">2</span>
            <h3 style="margin: 0;">Analyze</h3>
        </div>
        <p style="margin: 0;">Our system processes vaccination likelihood</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">3</span>
            <h3 style="margin: 0;">Recommend</h3>
        </div>
        <p style="margin: 0;">Get targeted campaign strategies</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# File Upload Section
# =========================
st.markdown("""
<div class="card">
    <h2>Upload Your Data</h2>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose survey data file (CSV format)", type=["csv"],
                           help="Upload your CSV file containing survey responses")

if uploaded is None:
    st.info("Please upload your survey data to begin analysis")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")

# =========================
# Data Preview
# =========================
st.markdown("""
<div class="card">
    <h3>Data Preview</h3>
</div>
""", unsafe_allow_html=True)
st.dataframe(df_raw.head())

# =========================
# Analysis Section
# =========================
st.markdown("""
<div class="card">
    <h2>Generate Predictions</h2>
</div>
""", unsafe_allow_html=True)

if st.button("Analyze Vaccination Likelihood", use_container_width=True):
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
        
        # =========================
        # Metrics
        # =========================
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        st.markdown("""
        <div class="highlight-box">
            <h3>Results Summary</h3>
        </div>
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

        # =========================
        # Visualization
        # =========================
        st.markdown("""
        <div class="card">
            <h3>Vaccination Likelihood Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Define color palette
        teal_colors = ['#008080', '#B0E0E6']
        
        # H1N1 pie chart
        ax[0].pie(
            [h1_vax_pct, 100 - h1_vax_pct], 
            labels=["Likely", "Unlikely"], 
            autopct='%1.1f%%',
            startangle=90,
            colors=teal_colors,
            explode=(0.05, 0),
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
        )
        ax[0].set_title("H1N1 Vaccination", fontweight='bold', fontsize=14, pad=20)
        
        # Seasonal pie chart
        ax[1].pie(
            [seas_vax_pct, 100 - seas_vax_pct], 
            labels=["Likely", "Unlikely"], 
            autopct='%1.1f%%',
            startangle=90,
            colors=teal_colors,
            explode=(0.05, 0),
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
        )
        ax[1].set_title("Seasonal Vaccination", fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        fig.patch.set_facecolor('white')
        
        st.pyplot(fig)

        # =========================
        # Recommendation prompt
        # =========================
        st.markdown("""
        <div class="highlight-box">
            <h3>Ready for Campaign Recommendations?</h3>
            <p>Visit the Recommendations page to see targeted strategies based on this analysis.</p>
        </div>
        """, unsafe_allow_html=True)