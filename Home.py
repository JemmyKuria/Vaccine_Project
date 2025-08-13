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
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .step-card {
        background-color: #e0f7fa;
        border-left: 4px solid #008080;
        border-radius: 8px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
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
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-card h2 {
        color: #008080;
        margin-top: 0.5rem;
    }
    .highlight-box {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-left: 4px solid #008080;
    }
    /* Teal Button Styling */
    .stButton > button {
        background-color: #008080 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #006666 !important;
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }
    .section-title {
        color: #008080;
        border-bottom: 2px solid #008080;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .data-preview {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .about-content {
        line-height: 1.6;
    }
    .about-content ul {
        padding-left: 1.5rem;
    }
    .about-content li {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="header">
    <h1>Vaccine Campaign Optimizer</h1>
    <p>Maximize vaccination rates through data-driven outreach strategies</p>
</div>
""", unsafe_allow_html=True)


# =========================
# About Section with Columns
# =========================
with st.container():
    st.markdown('<h2 class="section-title">About This Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-content">
        <p>This application helps public health officials and vaccine campaign managers optimize their outreach efforts through advanced data analysis and machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-content">
        <p>Our tool analyzes survey responses to help you focus your resources where they'll have the greatest impact, improving vaccination rates while reducing campaign costs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the features and benefits
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="highlight-box" style="height: 100%;">
            <h4>Key Features:</h4>
            <ul>
                <li><strong>Predict vaccination likelihood</strong> for different demographic groups</li>
                <li><strong>Identify key factors</strong> influencing vaccine acceptance</li>
                <li><strong>Generate targeted outreach</strong> recommendations</li>
                <li><strong>Visualize campaign performance</strong> metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box" style="height: 100%;">
            <h4>How It Benefits You:</h4>
            <ul>
                <li>Increase vaccination rates by 15-30% through targeted outreach</li>
                <li>Reduce campaign costs by focusing on high-impact groups</li>
                <li>Make data-driven decisions with clear visualizations</li>
                <li>Adapt strategies based on real population data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    

# =========================
# How It Works Section
# =========================
st.markdown("""
<h2 class="section-title">How It Works</h2>
""", unsafe_allow_html=True)

# Create columns for the step cards
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">1</span>
            <h3 style="margin: 0;">Upload Data</h3>
        </div>
        <p style="margin: 0;">Simply upload your CSV file containing survey responses. Our system accepts standard demographic and behavioral survey data.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">2</span>
            <h3 style="margin: 0;">Advanced Analysis</h3>
        </div>
        <p style="margin: 0;">Our machine learning models process the data to predict vaccination likelihood and identify key influencing factors.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="step-number">3</span>
            <h3 style="margin: 0;">Actionable Insights</h3>
        </div>
        <p style="margin: 0;">Receive targeted campaign strategies tailored to your specific population segments for maximum effectiveness.</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# File Upload Section
# =========================
st.markdown("""
<div class="card">
    <h2 class="section-title">Upload Your Data</h2>
    <p>To begin your analysis, upload your survey data in CSV format. Ensure your file includes demographic information and relevant survey responses.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose survey data file (CSV format)", type=["csv"],
                          help="Upload your CSV file containing demographic and survey response data")

if uploaded is None:
    st.info("ℹ️ Please upload your survey data to begin analysis. Sample data format: age, gender, education level, health conditions, etc.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")

# =========================
# Data Preview
# =========================
st.markdown("""
<div class="card">
    <h3 class="section-title">Data Preview</h3>
    <p>Here's a preview of your uploaded data. Ensure all relevant columns are properly formatted.</p>
</div>
""", unsafe_allow_html=True)

st.dataframe(df_raw.head().style.set_properties(**{'background-color': '#f8f9fa', 
                                                 'border': '1px solid #e0e0e0'}))

# =========================
# Analysis Section
# =========================
st.markdown("""
<div class="card">
    <h2 class="section-title">Generate Predictions</h2>
    <p>Click the button below to analyze your data and generate vaccination likelihood predictions. This process may take a few moments depending on your dataset size.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Analyze Vaccination Likelihood", use_container_width=True):
    with st.spinner("🔍 Analyzing data... This may take a moment for larger datasets"):
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
        st.success("🎉 Analysis complete! Visit the Recommendations page for detailed campaign strategies.")
        
        # =========================
        # Metrics
        # =========================
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        st.markdown("""
        <div class="highlight-box">
            <h3>Results Summary</h3>
            <p>Key metrics from your vaccination likelihood analysis:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Respondents</h4>
                <h2>{total:,}</h2>
                <p>individuals in your dataset</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>H1N1 Vaccine Likely</h4>
                <h2>{h1_vax_pct:.1f}%</h2>
                <p>predicted acceptance rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Seasonal Vaccine Likely</h4>
                <h2>{seas_vax_pct:.1f}%</h2>
                <p>predicted acceptance rate</p>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # Visualization
        # =========================
        st.markdown("""
        <div class="card">
            <h3 class="section-title">Vaccination Likelihood Distribution</h3>
            <p>Visual representation of predicted vaccination acceptance rates</p>
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
            <p>Our analysis has identified key patterns in your data. Visit the Recommendations page to see targeted outreach strategies tailored to your specific population segments.</p>
            <p>You'll receive guidance on:</p>
            <ul>
                <li>Which demographic groups to prioritize</li>
                <li>Effective messaging approaches</li>
                <li>Optimal communication channels</li>
                <li>Resource allocation strategies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .stButton>button {
        background-color: #008080 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #006666 !important;
        transform: scale(1.02);
    }</style>
        """, unsafe_allow_html=True)

    # Button to navigate to Recommendations page
    if st.button("🚀 Continue to Data Preview", use_container_width=True):
        st.session_state["clean_df"] = df_clean
        st.switch_page("pages/1Data_Preview.py")