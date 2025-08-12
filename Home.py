# 01_🏠_Home.py
import streamlit as st
import pandas as pd
from pipeline import preprocess, predict

# -------------------------------------------------
# PAGE CONFIGURATION (must be first Streamlit call)
# -------------------------------------------------
st.set_page_config(
    page_title="VaxPredict: Vaccine Recommendation Engine",
    page_icon="💉",
    layout="wide"
)

# -------------------------------------
# Custom CSS styling (shared across all pages)
# -------------------------------------
def inject_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
        }
        
        /* Header styling */
        .header {
            color: #2c3e50;
            padding-bottom: 1rem;
            border-bottom: 2px solid #3498db;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
        }
        
        /* About section styling */
        .about-section {
            background-color: #e8f4fc;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 2rem;
            border-left: 5px solid #3498db;
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-title {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        .metric-value {
            color: #2c3e50;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        /* Navigation sidebar */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
            background-color: #e8f4fc;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3498db;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------
# Home Page Function
# -------------------------------------
def home_page():
    inject_css()

    # Header with logo
    st.markdown("""
    <div class="header">
        <h1 style="display: flex; align-items: center;">
            <span style="color: #3498db;">💉</span>
            <span style="margin-left: 10px;">VaxPredict</span>
        </h1>
        <p style="color: #7f8c8d;">AI-powered vaccine recommendation engine for public health</p>
    </div>
    """, unsafe_allow_html=True)

    # About section
    with st.expander("ℹ️ About This App", expanded=True):
        st.markdown("""
        <div class="about-section">
            <h3 style="color: #2c3e50; margin-top: 0;">What does this app do?</h3>
            <p>VaxPredict helps public health professionals identify individuals who may be hesitant about vaccination and provides personalized recommendations to increase vaccine uptake.</p>
            
            <h4 style="color: #2c3e50;">Key Features:</h4>
            <ul>
                <li>Analyzes survey data to predict vaccine hesitancy</li>
                <li>Identifies high-risk demographic groups</li>
                <li>Detects psychological and behavioral barriers</li>
                <li>Generates personalized messaging recommendations</li>
                <li>Provides actionable insights for targeted interventions</li>
            </ul>
            
            <h4 style="color: #2c3e50;">How to use:</h4>
            <ol>
                <li>Upload your survey data (CSV format)</li>
                <li>Click "Process Data and Predict"</li>
                <li>Explore the results in the different tabs</li>
                <li>Use the recommendations to guide your outreach</li>
            </ol>
            
            <p style="font-style: italic; color: #7f8c8d;">
                This tool is designed for public health professionals and researchers working to improve vaccination rates.
                All analysis is performed locally - your data never leaves your computer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content card
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50; margin-top: 0;">Upload Your Survey Data</h3>
        <p>Get started by uploading your vaccination survey data in CSV format. The data should include demographic information and health behavior responses.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Please upload a CSV file to begin analysis")
        return None

    # Show raw data preview
    df_raw = pd.read_csv(uploaded)
    with st.expander("📋 Raw Data Preview (First 5 Rows)"):
        st.dataframe(df_raw.head())

    # Process and predict button
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h3 style="color: #2c3e50;">Ready to Analyze</h3>
        <p>Click the button below to process your data and generate predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Process Data and Predict", key="process_button"):
        with st.spinner("Analyzing data and generating predictions..."):
            # Clean and predict labels directly
            df_clean = preprocess(df_raw.copy())
            h1n1_label, seasonal_label = predict(df_clean)  # Expecting tuple of arrays/lists

            # Create final results dataframe
            results = df_raw.copy()
            results["h1n1_label"] = h1n1_label
            results["seasonal_label"] = seasonal_label

            # Store in session state
            st.session_state["results_df"] = results

            # Display success message
            st.success("✅ Analysis complete! Visit the other pages to explore results.")

            # Summary metrics
            total = len(results)
            h1_vax_pct = results["h1n1_label"].mean() * 100
            seas_vax_pct = results["seasonal_label"].mean() * 100

            st.markdown("""
            <div style="margin-top: 2rem;">
                <h3 style="color: #2c3e50;">Summary Statistics</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Respondents</div>
                    <div class="metric-value">{total:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">H1N1 Vaccination Likely</div>
                    <div class="metric-value">{h1_vax_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Seasonal Vaccination Likely</div>
                    <div class="metric-value">{seas_vax_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        return results

    return None

# -------------------------------------
# Run the home page
# -------------------------------------
if __name__ == "__main__":
    home_page()
