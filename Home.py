# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from annotated_text import annotated_text

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="VaxTrend - Vaccine Insights",
    page_icon="💉",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        /* General page background */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Headers */
        h1, h2, h3 {
            color: #2b6777;
        }
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #52ab98;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #2b6777;
            color: white;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #52ab98;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #40867c;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("💉 VaxTrend")
st.subheader("Vaccine Uptake Insights & Analysis")

annotated_text(
    ("Interactive", "", "#2b6777"),
    " dashboards to explore ",
    ("vaccine hesitancy", "", "#f4a261"),
    " and drive informed public health action."
)

# ---------- SIDEBAR ----------
st.sidebar.header("📊 Controls")
vaccine_type = st.sidebar.selectbox("Select Vaccine Type", ["H1N1", "Seasonal Flu", "COVID-19"])
year = st.sidebar.slider("Select Year", 2015, 2025, 2023)

# Dummy button
if st.sidebar.button("Apply Filters"):
    st.sidebar.success("Filters applied!")

# ---------- DATA (Demo) ----------
data = pd.DataFrame({
    "Age Group": ["18-24", "25-34", "35-44", "45-54", "55+"],
    "Vaccinated": [45, 60, 55, 65, 70]
})

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total Respondents", "5,432", "+3.5%")
col2.metric("Vaccination Rate", "62%", "+1.8%")
col3.metric("Hesitancy Rate", "38%", "-1.2%")

st.markdown("---")

# ---------- CHART ----------
fig = px.bar(
    data,
    x="Age Group",
    y="Vaccinated",
    text="Vaccinated",
    title=f"Vaccination Rate by Age Group ({vaccine_type} - {year})",
    color="Age Group",
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

st.plotly_chart(fig, use_container_width=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
