# pages/03_Preprocessing.py
import streamlit as st
import pandas as pd
from pipeline import preprocess

st.set_page_config(page_title="Pre-processing")
st.title("Data Cleaning & Pre-processing Report")

if "results_df" not in st.session_state:
    st.warning("Please upload a file on the Home page first.")
    st.stop()

df_raw = st.session_state["results_df"]


# 1. Missing-value summary BEFORE

st.subheader("1. Missing values BEFORE cleaning")
before = df_raw.isnull().sum()
before = before[before > 0].sort_values(ascending=False)
st.dataframe(before.rename("Missing Count"))


# 2. Clean

df_clean = preprocess(df_raw.copy())


# 3. Missing values AFTER

st.subheader("2. Missing values AFTER cleaning")
after = df_clean.isnull().sum()
after = after[after > 0].sort_values(ascending=False)
if after.empty:
    st.write("✅ No missing values remain.")
else:
    st.dataframe(after.rename("Missing Count"))


# 4. New / engineered columns
st.subheader("3. New / engineered columns")
explain = {
    "household_size": "Total people in household (adults + children)",
    "safe_behavior_score": "Sum of six safe-behaviour flags (0-6)",
    "doctor_recc_both": "Count of doctor recommendations (0-2)",
    "health_insurance": "1 = Yes, 0 = No, -1 = Unknown",
    "education": "Ordinal scale 0-3 (low to high)",
    "income_poverty": "Ordinal scale 0-2 (low to high)",
    "age_group": "Ordinal scale 0-4 (young to old)"
}
st.table(pd.Series(explain, name="Description"))


st.subheader("4. Cleaned Data Preview")
st.write("Preview (first 5 rows):")
st.dataframe(df_clean.head())


# 5. Download cleaned file

csv = df_clean.to_csv(index=False)
st.download_button(
    label="Download cleaned CSV",
    data=csv,
    file_name="cleaned_vaccine_data.csv",
    mime="text/csv"
)


# 6. Continue button

if st.button("Go to Predictions"):
    st.switch_page("pages/Predictions.py")