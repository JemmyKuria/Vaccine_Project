# pages/04_Predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pipeline import preprocess  # already handles cleaning & column order

st.set_page_config(page_title="Predictions", page_icon="🤖")
st.title("Prediction Results")

# -------------------------------------------------
# 1. Sanity check
# -------------------------------------------------
if "raw_df" not in st.session_state:
    st.warning("Please upload a file on the Home page first.")
    st.stop()

# -------------------------------------------------
# 2. Load model (cached so it runs once)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("multi_tuned_rf.pkl")

model = load_model()

# -------------------------------------------------
# 3. Clean & predict
# -------------------------------------------------
df_raw   = st.session_state["raw_df"]
df_clean = preprocess(df_raw.copy())
proba    = model.predict_proba(df_clean)          # list of two 2-D arrays
h1n1_prob      = proba[0][:, 1]
seasonal_prob  = proba[1][:, 1]

# -------------------------------------------------
# 4. Build results DataFrame
# -------------------------------------------------
results = df_raw.copy()
results["h1n1_prob"]     = h1n1_prob
results["seasonal_prob"] = seasonal_prob
results["h1n1_label"]    = (h1n1_prob >= 0.5).astype(int)
results["seasonal_label"]= (seasonal_prob >= 0.5).astype(int)

# -------------------------------------------------
# 5. KPI cards
# -------------------------------------------------
total_rows = len(results)
h1_high_risk = (results["h1n1_label"] == 0).sum()
seas_high_risk = (results["seasonal_label"] == 0).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total rows processed", f"{total_rows:,}")
col2.metric("High-risk (H1N1)", f"{h1_high_risk:,}")
col3.metric("High-risk (Seasonal)", f"{seas_high_risk:,}")

# -------------------------------------------------
# 6. Bar chart
# -------------------------------------------------
counts = pd.DataFrame({
    "H1N1": [results["h1n1_label"].sum(), total_rows-results["h1n1_label"].sum()],
    "Seasonal": [results["seasonal_label"].sum(), total_rows-results["seasonal_label"].sum()]
}, index=["Vaccinated", "Un-vaccinated"])

fig_bar, ax_bar = plt.subplots(figsize=(6,3))
counts.plot(kind="bar", ax=ax_bar)
ax_bar.set_title("Predicted uptake")
ax_bar.set_ylabel("Count")
st.pyplot(fig_bar)

# -------------------------------------------------
# 7. Pie charts
# -------------------------------------------------
fig_pie, axes = plt.subplots(1, 2, figsize=(8,3))
axes[0].pie(counts["H1N1"], labels=counts.index, autopct='%1.1f%%')
axes[0].set_title("H1N1")
axes[1].pie(counts["Seasonal"], labels=counts.index, autopct='%1.1f%%')
axes[1].set_title("Seasonal")
st.pyplot(fig_pie)

# -------------------------------------------------
# 8. Download CSV
# -------------------------------------------------
csv = results.to_csv(index=False)
st.download_button(
    label="Download predictions CSV",
    data=csv,
    file_name="vaccine_predictions.csv",
    mime="text/csv"
)