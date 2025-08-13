# pages/03_Preprocessing.py
import streamlit as st
import pandas as pd
from pipeline import preprocess
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Pre-processing", 
    page_icon="🧹",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
    /* Main styling */
    .header {
        background-color: #008080;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #008080;
    }
    .feature-card {
        background-color: #f0f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4C78A8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #e0f7fa;
        border-radius: 8px 8px 0 0;
        border: none !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #008080 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #008080 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #006666 !important;
        transform: scale(1.02);
    }
    .stSelectbox>div>div {
        border-color: #008080 !important;
    }
    .stDataFrame {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="header">
    <h1 style="margin:0; color:white;">🧹 Data Cleaning & Pre-processing Report</h1>
    <p style="margin:0; color:white; opacity:0.9;">Understand how your data was transformed for optimal analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("### Navigation")
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("Home.py")
    if st.button("Go to Predictions →", use_container_width=True):
        st.switch_page("pages/Predictions.py")
    st.markdown("---")
    
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#008080;">Current Data Shape</h4>
    """, unsafe_allow_html=True)
    if "results_df" in st.session_state:
        st.metric("Rows × Columns", 
                f"{st.session_state['results_df'].shape[0]} × {st.session_state['results_df'].shape[1]}",
                help="Dimensions of your processed dataset")
    else:
        st.warning("No data loaded")
    st.markdown("</div>", unsafe_allow_html=True)

# Check for data
if "results_df" not in st.session_state:
    st.warning("Please upload a file on the Home page first.")
    st.stop()

df_raw = st.session_state["results_df"]

# Main content
tab1, tab2 = st.tabs(["🔍 Data Cleaning Report", "📊 Feature Engineering"])

with tab1:
    # 1. Missing-value summary BEFORE
    st.markdown("""
    <h3 style="color:#008080; border-bottom:2px solid #008080; padding-bottom:8px;">
        Missing Values Analysis
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color:#008080;">Before Cleaning</h4>
        </div>
        """, unsafe_allow_html=True)
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
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color:#008080;">After Cleaning</h4>
        </div>
        """, unsafe_allow_html=True)
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
    st.markdown("""
    <h3 style="color:#008080; border-bottom:2px solid #008080; padding-bottom:8px; margin-top:20px;">
        Cleaned Data Preview
    </h3>
    """, unsafe_allow_html=True)
    st.dataframe(df_clean.head().style.set_properties(**{
        'background-color': '#f8f9fa',
        'border': '1px solid #e0e0e0'
    }), use_container_width=True)

with tab2:
    st.markdown("""
    <h3 style="color:#4C78A8; border-bottom:2px solid #4C78A8; padding-bottom:8px;">
        New Features Created
    </h3>
    """, unsafe_allow_html=True)
    
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
            "purpose": "Allows models to understand age as an ordered category rather than text",
            "color": "#4C78A8"
        },
        "education": {
            "description": "Education level converted to numeric scale",
            "mapping": {
                "0": "< 12 Years",
                "1": "High School (12 Years)",
                "2": "Some College",
                "3": "College Graduate"
            },
            "purpose": "Captures education progression as ordinal values",
            "color": "#54A24B"
        },
        "income_poverty": {
            "description": "Income level converted to simple scale",
            "mapping": {
                "0": "Below Poverty",
                "1": "<= $75K (Above Poverty)",
                "2": "> $75K"
            },
            "purpose": "Simplifies complex income categories into meaningful tiers",
            "color": "#EECA3B"
        },
        "health_insurance": {
            "description": "Insurance status simplified",
            "mapping": {
                "1": "Yes",
                "0": "No",
                "-1": "Unknown"
            },
            "purpose": "Handles missing values while preserving information",
            "color": "#F58518"
        },
        "household_size": {
            "description": "Calculated total people in home",
            "formula": "Adults + Children",
            "purpose": "Combines two related columns into one informative feature",
            "color": "#72B7B2"
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
            "purpose": "Creates single measure of COVID-safe behaviors",
            "color": "#B279A2"
        },
        "doctor_recc_both": {
            "description": "Combined doctor recommendations",
            "calculation": "H1N1 recommendation + Seasonal recommendation",
            "scale": "0-2 (number of recommendations received)",
            "purpose": "Shows strength of medical advice received",
            "color": "#FF9DA6"
        }
    }
    
    # Display as expandable cards
    cols = st.columns(2, gap="large")
    for i, (feature, details) in enumerate(transformation_details.items()):
        with cols[i % 2]:
            with st.expander(f"🔧 {feature.replace('_', ' ').title()}", expanded=True):
                st.markdown(f"""
                <div style="background-color:{details.get('color', '#f0f9f9')}20; 
                            padding:12px; border-radius:8px; border-left:4px solid {details.get('color', '#4C78A8')};">
                    <p style="margin:0; font-weight:bold; color:{details.get('color', '#4C78A8')};">{details['description']}</p>
                    <p style="margin:4px 0 0 0; font-size:0.9em; color:#666;">{details['purpose']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'mapping' in details:
                    st.markdown("**Category Mapping:**")
                    for num, text in details['mapping'].items():
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; margin:4px 0;">
                            <div style="background-color:{details.get('color', '#4C78A8')}30; 
                                        border-radius:4px; padding:2px 8px; margin-right:8px;">
                                {num}
                            </div>
                            <div>→ {text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if 'formula' in details:
                    st.markdown(f"**Calculation:** `{details['formula']}`")
                
                if 'components' in details:
                    st.markdown("**Combines:**")
                    for item in details['components']:
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; margin:4px 0;">
                            <div style="width:6px; height:6px; background-color:{details.get('color', '#4C78A8')}; 
                                        border-radius:50%; margin-right:8px;"></div>
                            <div>{item}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show stats if column exists
                if feature in df_clean.columns:
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unique Values", df_clean[feature].nunique())
                    with col2:
                        st.metric("Missing Values", df_clean[feature].isnull().sum())
    
    # Distribution visualization
    st.markdown("""
    <h3 style="color:#4C78A8; border-bottom:2px solid #4C78A8; padding-bottom:8px; margin-top:20px;">
        Feature Distributions
    </h3>
    """, unsafe_allow_html=True)
    
    selected_feature = st.selectbox(
        "Select feature to explore:",
        options=list(transformation_details.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="feature_selector"
    )
    
    if selected_feature in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[selected_feature]):
            fig = px.histogram(
                df_clean,
                x=selected_feature,
                title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
                color_discrete_sequence=[transformation_details[selected_feature].get('color', '#4C78A8')],
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
                color_discrete_sequence=[transformation_details[selected_feature].get('color', '#4C78A8')]
            )
        
        # Add reference lines for ordinal features
        if selected_feature in ['age_group', 'education', 'income_poverty']:
            for val, label in transformation_details[selected_feature]['mapping'].items():
                fig.add_annotation(
                    x=float(val),
                    y=0,
                    text=label,
                    showarrow=False,
                    yshift=-40,
                    font=dict(color=transformation_details[selected_feature].get('color', '#4C78A8')))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer actions
st.divider()
col1, col2 = st.columns(2, gap="large")
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