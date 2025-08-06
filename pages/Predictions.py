import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Predictions", layout="wide")
st.title("Vaccination Prediction Dashboard")

# 1. Data Loading

if "results_df" not in st.session_state:
    st.warning("Please process data on the Home page first.")
    st.stop()

results = st.session_state["results_df"]

# 2. Key Metrics Dashboard

col1, col2, col3 = st.columns(3)
col1.metric("Total Respondents", f"{len(results):,}")
col2.metric("H1N1 Vaccine Non Takers",
           f"{(results['h1n1_label'] == 0).sum():,}",
           help="Number of respondents predicted not to get the H1N1 vaccine")
col3.metric("Seasonal Vaccine Non Takers",
           f"{(results['seasonal_label'] == 0).sum():,}",
           help="Number of respondents predicted not to get the seasonal vaccine")

# 3. Demographic Breakdown

st.subheader("Demographic Analysis")

# Get categorical columns
categorical_cols = results.select_dtypes(include=['object', 'category']).columns.tolist()
target_cols = ['h1n1_label', 'seasonal_label']


# Demographic selector
demo_col = st.selectbox(
    "Select demographic factor:",
    options=categorical_cols,
    index=categorical_cols.index('age_group') if 'age_group' in categorical_cols else 0
)

# Create visualization
demo_data = results.groupby(demo_col)[target_cols].apply(lambda x: (x == 0).sum()).reset_index()

fig = px.bar(
    demo_data,
    x=demo_col,
    y=target_cols,
    title=f"Count of Respondents Not Likely to Get Vaccine by {demo_col.replace('_', ' ').title()}",
    labels={'value': 'Count', 'variable': 'Vaccine'},
    barmode='group',
    color_discrete_sequence=['#636EFA', '#EF553B']  # Blue for H1N1, Red for Seasonal
)

fig.update_layout(
    yaxis_tickformat=',',
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 4. Data Export

st.subheader("Data Export")
csv = results.to_csv(index=False)
st.download_button(
    label="Download Predictions (CSV)",
    data=csv,
    file_name="vaccine_predictions.csv",
    mime="text/csv"
)