import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Vaccine Dashboard", layout="wide")
st.title("📊 Advanced Vaccination Prediction Dashboard")

# 1. Data Loading
if "results_df" not in st.session_state:
    st.warning("Please process data on the Home page first.")
    st.stop()

results = st.session_state["results_df"]

# 2. Key Metrics Dashboard
st.header("📈 Key Metrics")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total Respondents", f"{len(results):,}")
metric2.metric("H1N1 Non-Takers", f"{(results['h1n1_label'] == 0).sum():,}", 
               help="Predicted not to get H1N1 vaccine")
metric3.metric("Seasonal Non-Takers", f"{(results['seasonal_label'] == 0).sum():,}",
               help="Predicted not to get seasonal vaccine")
metric4.metric("Double Non-Takers", 
               f"{((results['h1n1_label'] == 0) & (results['seasonal_label'] == 0)).sum():,}",
               help="Predicted to take neither vaccine")

# 3. Main Dashboard Tabs
tab1, tab2= st.tabs(["📊 Demographics", "📤 Export"])

with tab1:  # Demographic Analysis
    st.subheader("Population Breakdown")
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        demo_factor = st.selectbox(
            "Select demographic factor:",
            options=results.select_dtypes(include=['object', 'category']).columns,
            index=0
        )
    
    with demo_col2:
        view_type = st.radio("View:", ["Count", "Percentage"], horizontal=True)
    
    # Demographic bar chart
    demo_data = results.groupby(demo_factor)[['h1n1_label', 'seasonal_label']].mean().reset_index()
    if view_type == "Count":
        demo_data = results.groupby(demo_factor)[['h1n1_label', 'seasonal_label']]\
                          .apply(lambda x: (x == 0).sum()).reset_index()
    
    fig = px.bar(
        demo_data,
        x=demo_factor,
        y=['h1n1_label', 'seasonal_label'],
        barmode='group',
        title=f"Vaccine Hesitancy by {demo_factor.replace('_', ' ').title()}",
        labels={'value': 'Count' if view_type == "Count" else 'Percentage', 'variable': 'Vaccine'},
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Demographic pie charts
    st.subheader("Population Composition")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(results, names=demo_factor, title=f"Distribution by {demo_factor}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        selected_vaccine = st.selectbox("Show uptake for:", ['h1n1_label', 'seasonal_label'])
        fig = px.pie(results, names=selected_vaccine, title=f"{selected_vaccine.replace('_', ' ').title()} Distribution")
        st.plotly_chart(fig, use_container_width=True)



with tab2:  # Data Export
    st.subheader("Export Results")
    
    with st.expander("📄 Preview Data"):
        st.dataframe(results.head(100))
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Full Data (CSV)",
            data=results.to_csv(index=False),
            file_name="vaccine_predictions_full.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download Filtered Data (CSV)",
            data=results[['respondent_id', 'h1n1_label', 'seasonal_label']].to_csv(index=False),
            file_name="vaccine_predictions_minimal.csv",
            mime="text/csv"
        )
    
