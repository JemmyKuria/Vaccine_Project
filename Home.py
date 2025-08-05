# 01_Home.py
import streamlit as st
import pandas as pd
from pipeline import preprocess, predict

st.set_page_config(page_title="Home", page_icon="🏠")
st.title("Public-Health Vaccine Dashboard")

# -------------------------------------------------
# 1. File uploader
# -------------------------------------------------
uploaded = st.file_uploader("Upload survey CSV file", type=["csv"])
if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded)
st.write("Raw preview (first 5 rows):")
st.dataframe(df_raw.head())

# -------------------------------------------------
# 2. Two buttons
# -------------------------------------------------
col_a, col_b = st.columns(2)

if col_a.button("Continue"):
    df_clean = preprocess(df_raw.copy())
    st.session_state["df_clean"] = df_clean
    st.success("Data cleaned and stored.")


    h1n1_prob, seasonal_prob = predict(st.session_state["df_clean"])
    st.session_state["raw_df"]        = df_raw
    st.session_state["h1n1_prob"]     = h1n1_prob
    st.session_state["seasonal_prob"] = seasonal_prob

    total = len(df_raw)
    h1_unvax_pct  = (h1n1_prob < 0.5).mean() * 100
    seas_unvax_pct = (seasonal_prob < 0.5).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", f"{total:,}")
    col2.metric("H1N1 un-vaccinated", f"{h1_unvax_pct:.1f} %")
    col3.metric("Seasonal un-vaccinated", f"{seas_unvax_pct:.1f} %")

    st.success("Predictions ready! Visit Explore or Recommendation pages.")