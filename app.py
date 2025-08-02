import streamlit as st
from pipeline.preprocessing import DataPreprocessor
from pipeline.eda import Visualizer
import pandas as pd

# Streamlit app configuration
st.set_page_config(page_title="Survey Data Processor", layout="wide")

# Title and file uploader
st.title("📊 Survey Data Analysis Pipeline")
uploaded_file = st.file_uploader("Upload your survey data (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and process data
        with st.spinner('Processing data...'):
            raw_data = pd.read_csv(uploaded_file)
            
            preprocessed_data = (
                DataPreprocessor(raw_data)
                .validate_data()
                .drop_unused_columns()
                .handle_missing_values()
                .engineer_features()
                .encode_categoricals()
                .get_processed_data()
                
            )
            
            # Store in session state for multi-page access
            st.session_state['preprocessed_data'] = preprocessed_data
            
        # Display success and processed data
        st.success("✅ Data preprocessing completed successfully!")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Processed Data", "Exploratory Analysis"])
        
        with tab1:
            st.subheader("Processed Data Preview")
            st.dataframe(preprocessed_data.head())
            st.write(f"Shape: {preprocessed_data.shape}")
            
            # Show data summary
            if st.checkbox("Show data summary"):
                st.write(preprocessed_data.describe(include='all'))
        
        with tab2:
            st.subheader("Exploratory Data Analysis")
            visualizer = Visualizer(preprocessed_data)
            
            # Vaccination distribution
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(visualizer.plot_vaccination_distribution())
            
            # Correlation heatmap
            with col2:
                st.pyplot(visualizer.plot_correlation_heatmap())
            
            # Feature analysis
            st.subheader("Feature Analysis")
            selected_feature = st.selectbox(
                "Select a feature to analyze:",
                options=[col for col in preprocessed_data.columns 
                        if col not in ['h1n1_vaccine', 'risk_score']]
            )
            
            if selected_feature:
                try:
                    st.pyplot(visualizer.plot_feature_vs_target(selected_feature))
                except Exception as e:
                    st.warning(f"Could not plot {selected_feature}: {str(e)}")
            
            # Risk score distribution (if available)
            if 'risk_score' in preprocessed_data.columns:
                st.subheader("Risk Score Analysis")
                st.pyplot(visualizer.plot_risk_score_distribution())
        
        # Navigation to analysis page
        st.markdown("---")
        if st.button("Continue to Advanced Analysis"):
            st.switch_page("pages/analysis.py")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.error("Please check your data format and try again.")
else:
    st.info("ℹ️ Please upload a CSV file to begin analysis.")