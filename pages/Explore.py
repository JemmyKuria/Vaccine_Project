# pages/02_Explore.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pipeline import preprocess   # same cleaning steps you used

st.set_page_config(page_title="Explore", page_icon="📊")
st.title("Explore the Data")

# -------------------------------------------------
# 1. Check upload
# -------------------------------------------------
if "raw_df" not in st.session_state:
    st.info("Please upload a file on the Home page first.")
    st.stop()

df_raw = st.session_state["raw_df"]

# -------------------------------------------------
# 2. Clean once and cache
# -------------------------------------------------

 # Avoids re-running every time the user changes a sidebar filter
@st.cache_data
def get_clean():
    return preprocess(df_raw.copy())

df_clean = get_clean()

# -------------------------------------------------
# 3. Sidebar filters (simple multiselects)
# -------------------------------------------------
age_choices = st.sidebar.multiselect(
    "Age groups:",
    options=df_raw["age_group"].dropna().unique(),
    default=df_raw["age_group"].dropna().unique()
)

inc_choices = st.sidebar.multiselect(
    "Income level:",
    options=df_raw["income_poverty"].dropna().unique(),
    default=df_raw["income_poverty"].dropna().unique()
)

# filter
mask = (
    (df_raw["age_group"].isin(age_choices)) &
    (df_raw["income_poverty"].isin(inc_choices))
)
df_plot = df_raw[mask].copy()

# -------------------------------------------------
# 4. Three quick charts (Seaborn style you used)
# -------------------------------------------------
st.subheader("Age group vs H1N1 concern")
fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.countplot(data=df_plot, x="age_group", hue="h1n1_concern", ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig1)

st.subheader("Income vs H1N1 concern")
fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.boxplot(data=df_plot, x="income_poverty", y="h1n1_concern", ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig2)

st.subheader("Doctor recommendation vs opinion")
if "doctor_recc_h1n1" in df_plot.columns and "opinion_h1n1_vacc_effective" in df_plot.columns:
    df_heat = df_plot[["doctor_recc_h1n1", "opinion_h1n1_vacc_effective"]].copy()
    df_heat["doctor_recc_h1n1"] = df_heat["doctor_recc_h1n1"].astype(int)
    df_heat["opinion_h1n1_vacc_effective"] = df_heat["opinion_h1n1_vacc_effective"].astype(int)
    corr = df_heat.corr()
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

st.caption("Charts use cleaned survey data; no model predictions displayed here.")