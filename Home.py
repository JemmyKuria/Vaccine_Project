import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pipeline import preprocess, predict 

# Page Configuration
st.set_page_config(
    page_title="Vaccine Campaign Optimizer", 
    page_icon="💉",
    layout="wide"
)

# Header
st.title("Vaccine Campaign Optimizer")
st.markdown("---")

# About This Tool Section
st.header("About This Tool")
st.markdown("""
This application helps **public health officials** and **vaccine campaign managers** optimize their outreach efforts by:

1. Identifying populations with low vaccination likelihood  
2. Analyzing key behavioral and demographic factors affecting vaccine uptake  
3. Recommending targeted messaging strategies for different audience segments  
""")

st.subheader("How It Works")
st.markdown("""
1. **Upload** your survey data (CSV format)  
2. **Analyze** the predictions to understand vaccination likelihood  
3. **Explore** the recommendations page for targeted campaign strategies  
""")

st.caption("Designed for Ministries of Health and public health organizations to maximize vaccine campaign effectiveness.")
st.markdown("---")

# File Upload Section
st.header("1. Upload Survey Data")
uploaded = st.file_uploader("Choose your survey data file (CSV format)", type=["csv"])
    
if uploaded is None:
    st.info("Please upload your survey data to begin analysis")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")
    
# Data preview
st.subheader("Data Preview")
st.dataframe(df_raw.head())
st.markdown("---")

# Analysis Section
st.header("2. Generate Predictions")
    
if st.button("Analyze Vaccination Likelihood", type="primary"):
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
        col1.metric("Total Respondents", f"{total:,}")
        col2.metric("H1N1 Vaccine Likely", f"{h1_vax_pct:.1f}%")
        col3.metric("Seasonal Vaccine Likely", f"{seas_vax_pct:.1f}%")

        # Visualization
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