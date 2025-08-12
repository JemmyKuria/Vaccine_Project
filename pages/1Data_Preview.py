# pages/03_Preprocessing.py
import streamlit as st
import pandas as pd
from pipeline import preprocess
import plotly.express as px

# Page Config
st.set_page_config(page_title="Pre-processing", layout="wide")
st.title("🧹 Data Cleaning & Pre-processing Report")

# Sidebar for navigation
with st.sidebar:
    st.markdown("### Navigation")
    if st.button("← Back to Home"):
        st.switch_page("Home.py")
    if st.button("Go to Predictions →"):
        st.switch_page("pages/Predictions.py")
    st.markdown("---")
    st.markdown("**Current Data Shape:**")
    if "results_df" in st.session_state:
        st.write(f"{st.session_state['results_df'].shape[0]} rows × {st.session_state['results_df'].shape[1]} columns")

# Check for data
if "results_df" not in st.session_state:
    st.warning("Please upload a file on the Home page first.")
    st.stop()

df_raw = st.session_state["results_df"]

# Main content
tab1, tab2 = st.tabs(["🔍 Data Cleaning Report", "📊 New Features"])

with tab1:
    # 1. Missing-value summary BEFORE
    st.subheader("Missing Values Analysis", divider="blue")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Cleaning**")
        before = df_raw.isnull().sum()
        before = before[before > 0].sort_values(ascending=False)
        if not before.empty:
            st.dataframe(
                before.rename("Missing Count"), 
                use_container_width=True,
                height=min(300, len(before) * 35 + 3)
            )
        else:
            st.success("✅ No missing values found in raw data")
    
    # 2. Clean data
    df_clean = preprocess(df_raw.copy())
    
    with col2:
        st.markdown("**After Cleaning**")
        after = df_clean.isnull().sum()
        after = after[after > 0].sort_values(ascending=False)
        if not after.empty:
            st.dataframe(
                after.rename("Missing Count"), 
                use_container_width=True,
                height=min(300, len(after) * 35 + 3))
        else:
            st.success("✅ All missing values handled")

    # Data preview
    st.subheader("Cleaned Data Preview", divider="blue")
    st.dataframe(df_clean.head(), use_container_width=True)

with tab2:
    st.subheader("New Features Created", divider="green")
    
    # Detailed explanations for each transformation
    transformation_details = {
        "age_group": {
            "description": "Age converted to ordinal scale (young to old)",
            "mapping": {
                "0": "18-34 Years",
                "1": "35-44 Years",
                "2": "45-54 Years", 
                "3": "55-64 Years",
                "4": "65+ Years"
            },
            "purpose": "Allows models to understand age as an ordered category rather than text"
        },
        "education": {
            "description": "Education level converted to numeric scale",
            "mapping": {
                "0": "< 12 Years",
                "1": "High School (12 Years)",
                "2": "Some College",
                "3": "College Graduate"
            },
            "purpose": "Captures education progression as ordinal values"
        },
        "income_poverty": {
            "description": "Income level converted to simple scale",
            "mapping": {
                "0": "Below Poverty",
                "1": "<= $75K (Above Poverty)",
                "2": "> $75K"
            },
            "purpose": "Simplifies complex income categories into meaningful tiers"
        },
        "health_insurance": {
            "description": "Insurance status simplified",
            "mapping": {
                "1": "Yes",
                "0": "No",
                "-1": "Unknown"
            },
            "purpose": "Handles missing values while preserving information"
        },
        "household_size": {
            "description": "Calculated total people in home",
            "formula": "Adults + Children",
            "purpose": "Combines two related columns into one informative feature"
        },
        "safe_behavior_score": {
            "description": "Composite safety practices score",
            "components": [
                "Avoided crowds",
                "Wore face masks",
                "Washed hands frequently",
                "Limited gatherings",
                "Reduced outside activities",
                "Avoided face touching"
            ],
            "scale": "0-6 (sum of safety practices)",
            "purpose": "Creates single measure of COVID-safe behaviors"
        },
        "doctor_recc_both": {
            "description": "Combined doctor recommendations",
            "calculation": "H1N1 recommendation + Seasonal recommendation",
            "scale": "0-2 (number of recommendations received)",
            "purpose": "Shows strength of medical advice received"
        }
    }
    
    # Display as expandable cards
    cols = st.columns(2)
    for i, (feature, details) in enumerate(transformation_details.items()):
        with cols[i % 2]:
            with st.expander(f"🔧 {feature.replace('_', ' ').title()}", expanded=True):
                st.markdown(f"**{details['description']}**")
                st.caption(f"*Purpose:* {details['purpose']}")
                
                if 'mapping' in details:
                    st.markdown("**Category Mapping:**")
                    for num, text in details['mapping'].items():
                        st.write(f"{num} → {text}")
                
                if 'formula' in details:
                    st.markdown(f"**Calculation:** `{details['formula']}`")
                
                if 'components' in details:
                    st.markdown("**Combines:**")
                    for item in details['components']:
                        st.write(f"- {item}")
                
                # Show stats if column exists
                if feature in df_clean.columns:
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unique Values", df_clean[feature].nunique())
                    with col2:
                        st.metric("Missing Values", df_clean[feature].isnull().sum())
    
    # Distribution visualization
    st.subheader("Feature Distributions", divider="green")
    selected_feature = st.selectbox(
        "Select feature to explore:",
        options=list(transformation_details.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_feature in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[selected_feature]):
            fig = px.histogram(
                df_clean,
                x=selected_feature,
                title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
                color_discrete_sequence=['#4C78A8'],
                nbins=min(20, df_clean[selected_feature].nunique()),
                labels={'value': transformation_details[selected_feature]['description']}
            )
        else:
            fig = px.bar(
                df_clean[selected_feature].value_counts().reset_index(),
                x='index',
                y=selected_feature,
                title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
                color=selected_feature,
                color_continuous_scale='Teal'
            )
        
        # Add reference lines for ordinal features
        if selected_feature in ['age_group', 'education', 'income_poverty']:
            for val, label in transformation_details[selected_feature]['mapping'].items():
                fig.add_annotation(
                    x=float(val),
                    y=0,
                    text=label,
                    showarrow=False,
                    yshift=-40
                )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer actions
st.divider()
col1, col2 = st.columns(2)
with col1:
    csv = df_clean.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="cleaned_vaccine_data.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    if st.button("🚀 Continue to Predictions", use_container_width=True):
        st.session_state["clean_df"] = df_clean
        st.switch_page("pages/Predictions.py")