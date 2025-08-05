# pages/05_Recommendations.py
import streamlit as st
import pandas as pd
import numpy as np
from pipeline import preprocess, predict  # already handles cleaning

st.set_page_config(page_title="Recommendations", page_icon="📋")
st.title("Actionable Outreach Plan")

# -------------------------------------------------
# 1. Guard clause
# -------------------------------------------------
if "raw_df" not in st.session_state:
    st.warning("Please upload a file on the Home page first.")
    st.stop()

df_raw   = st.session_state["raw_df"]
df_clean = preprocess(df_raw.copy())
# -------------------------------------------------
# 2. Predict probabilities
h1n1_prob, seasonal_prob = predict(df_clean)
# -------------------------------------------------
# 3. Build results DataFrame
# -------------------------------------------------
results = df_raw.copy()
results["h1n1_label"]    = (h1n1_prob < 0.30).astype(int)   # 0 = non-taker
results["seasonal_label"]= (seasonal_prob < 0.30).astype(int)   # 0 = non-taker


# H1N1 non-takers (all features + labels)
h1_out = results[results["h1n1_label"] == 0].copy()

# Seasonal non-takers (all features + labels)
seas_out = results[results["seasonal_label"] == 0].copy()

# Download
st.download_button("Download H1N1 non-takers", 
                    h1_out.to_csv(index=False),
                   file_name="h1n1_non_takers.csv")
st.download_button("Download Seasonal non-takers",
                    seas_out.to_csv(index=False),
                   file_name="seasonal_non_takers.csv")