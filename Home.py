# 01_Home.py
import streamlit as st
import pandas as pd
from pipeline import preprocess, predict
import matplotlib.pyplot as plt

# Custom CSS
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .stFileUploader {
        background-color: #f2f2f2;
        padding: 10px;
        border-radius: 5px;
    }
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }
    .stMetric {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set page configuration
st.set_page_config(page_title="Home", page_icon="🏠")
st.title("Public-Health Vaccine Predictor")

# Add a logo or icon
#st.image("path_to_logo.png", width=100)

# 1. File uploader
with st.expander("Upload Data"):
    uploaded = st.file_uploader("Upload survey CSV file", type=["csv"])
    if uploaded is None:
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.write("Raw preview (first 5 rows):")
    st.dataframe(df_raw.head())

# 2. Process and Predict
with st.expander("Process Data and Predict"):
    if st.button("Process Data and Predict", help="Click to process and predict vaccination likelihood"):
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
        st.subheader("Summary Metrics")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # H1N1 vaccination likelihood
        ax[0].pie([h1_vax_pct, 100 - h1_vax_pct], labels=["Likely", "Unlikely"], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#f44336'])
        ax[0].set_title("H1N1 Vaccination Likelihood")

        # Seasonal vaccination likelihood
        ax[1].pie([seas_vax_pct, 100 - seas_vax_pct], labels=["Likely", "Unlikely"], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#f44336'])
        ax[1].set_title("Seasonal Vaccination Likelihood")

        st.pyplot(fig)

# Add a footer
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <p>Developed by Your Team Name</p>
    </div>
    """,
    unsafe_allow_html=True,
)