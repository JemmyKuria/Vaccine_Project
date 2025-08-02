import streamlit as st
import pandas as pd
import joblib

from pipeline.eda import Visualizer
from pipeline.preprocessing import preprocess_data  # your custom cleaning function
from pipeline.modelling import predict_vaccination  # wraps model.predict()

# Load model once
#model = joblib.load("models/model.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Analysis"])

# Upload CSV
st.title("🧬 Vaccination Prediction Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("✅ Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocess and predict
    try:
        cleaned_df = preprocess_data(df)
        predictions = predict_vaccination(cleaned_df, model)

        df['Predicted_vaccinated'] = predictions

        st.success("✅ Predictions complete!")

        # Dashboard tab
        if page == "Dashboard":
            st.subheader("📊 Vaccination Summary")
            vaccinated_count = df['Predicted_vaccinated'].value_counts().to_dict()
            st.write(f"**Will take vaccine:** {vaccinated_count.get(1, 0)}")
            st.write(f"**Will not take vaccine:** {vaccinated_count.get(0, 0)}")

        # Analysis tab
        elif page == "Analysis":
            st.subheader("📈 Data Analysis")
            viz = Visualizer(df)

            st.pyplot(viz.plot_vaccination_distribution())
            st.pyplot(viz.plot_correlation_heatmap())

            st.pyplot(viz.plot_feature_vs_target("age_group"))
            st.pyplot(viz.plot_feature_vs_target("education"))
            
    except Exception as e:
        st.error(f"Error during processing: {e}")

else:
    st.info("👆 Upload a CSV file to get started.")
