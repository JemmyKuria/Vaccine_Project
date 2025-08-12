import streamlit as st
import pandas as pd
from pipeline import preprocess, predict
import matplotlib.pyplot as plt

# Custom CSS to remove expander icons and enhance UI
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 8px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stFileUploader {
        background-color: #f2f2f2;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Remove expander icons */
    .stExpander .st-emotion-cache-1qrv0ga {
        display: none;
    }
    /* Custom section styling */
    .model-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set page configuration
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.title("Public Health Vaccine Predictor")

# Header with logo placeholder
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://via.placeholder.com/150x80?text=Vaccine+AI", width=150)
with col2:
    st.markdown("<h2 style='color: #2c3e50; margin-top: 20px;'>Predicting Vaccination Likelihood</h2>", unsafe_allow_html=True)

# About the Model section
with st.container():
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
            <p>The model considers factors like:
            <ul>
                <li>Perceived vaccine effectiveness</li>
                <li>Previous vaccination history</li>
                <li>Doctor recommendations</li>
                <li>Demographic information</li>
                <li>Health insurance status</li>
            </ul>
            </p>
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

# 1. File uploader
st.markdown("### Upload Your Survey Data")
with st.expander("Upload Data", expanded=True):
    uploaded = st.file_uploader("Upload survey CSV file", type=["csv"])
    if uploaded is None:
        st.info("Please upload a CSV file to get started")
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.write("Raw preview (first 5 rows):")
    st.dataframe(df_raw.head())

# 2. Process and Predict
st.markdown("### Generate Predictions")
with st.expander("Process Data and Predict", expanded=True):
    if st.button("Process Data and Predict", help="Click to process and predict vaccination likelihood"):
        with st.spinner('Processing data and generating predictions...'):
            # Clean and predict labels directly
            df_clean = preprocess(df_raw.copy())
            h1n1_label, seasonal_label = predict(df_clean)  # Returns 0/1 labels

            # Create final results dataframe
            results = df_raw.copy()
            results["h1n1_label"] = h1n1_label
            results["seasonal_label"] = seasonal_label

            # Store in session state
            st.session_state["results_df"] = results
            st.success("Predictions ready! Visit other pages to explore results.")

            # Display summary metrics
            total = len(results)
            h1_vax_pct = results["h1n1_label"].mean() * 100
            seas_vax_pct = results["seasonal_label"].mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total respondents", f"{total:,}")
            col2.metric("H1N1 vaccination likely", f"{h1_vax_pct:.1f}%")
            col3.metric("Seasonal vaccination likely", f"{seas_vax_pct:.1f}%")

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

