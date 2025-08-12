import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pipeline import preprocess, predict 

# Page Configuration
st.set_page_config(
    page_title="Vaccine Prediction Dashboard", 
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("Public Health Vaccine Predictor")
st.markdown("""
    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; margin-bottom:20px">
        <h3 style="color:#2e7d32; margin:0">Upload survey data to predict vaccination likelihood</h3>
    </div>
""", unsafe_allow_html=True)

# 1. File Upload Section
with st.expander("📁 Upload Survey Data", expanded=True):
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload survey data in CSV format")
    
    if uploaded is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.success(f"✅ Successfully uploaded {len(df_raw)} records")
    
    with st.expander("View Raw Data Preview"):
        st.dataframe(df_raw.head().style.highlight_max(axis=0, color='#c8e6c9'))

# 2. Processing Section
st.markdown("---")
if st.button("🚀 Process Data & Generate Predictions", type="primary"):
    with st.spinner("Processing data and generating predictions..."):
        # Clean and predict
        df_clean = preprocess(df_raw.copy())
        h1n1_label, seasonal_label = predict(df_clean)
        
        # Create final results
        results = df_raw.copy()
        results["h1n1_label"] = h1n1_label
        results["seasonal_label"] = seasonal_label
        
        # Store in session state
        st.session_state["results_df"] = results
        
        # Success message
        st.balloons()
        st.success("""
        ### Predictions Complete!
        Visit the **Recommendation** page to explore insights and messaging strategies.
        """)
        
        # Metrics Display
        st.markdown("---")
        st.subheader("📊 Prediction Summary")
        
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        # Modern metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color:#2e7d32">Total Respondents</h3>
                <h2>{:,}</h2>
            </div>
            """.format(total), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color:#2e7d32">H1N1 Vaccination Likely</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(h1_vax_pct), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color:#2e7d32">Seasonal Vaccination Likely</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(seas_vax_pct), unsafe_allow_html=True)
        
        # Interactive visualization
        st.markdown("---")
        st.subheader("📈 Vaccination Likelihood Distribution")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Matplotlib Charts", "Interactive Plotly Charts"])
        
        with tab1:
            # Matplotlib pie charts
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # H1N1 pie chart
            ax[0].pie([h1_vax_pct, 100 - h1_vax_pct], 
                     labels=["Likely", "Unlikely"], 
                     autopct='%1.1f%%',
                     startangle=90,
                     colors=['#81c784', '#ff8a65'],
                     explode=(0.1, 0),
                     textprops={'fontsize': 12})
            ax[0].set_title("H1N1 Vaccination", pad=20, fontweight='bold')
            
            # Seasonal pie chart
            ax[1].pie([seas_vax_pct, 100 - seas_vax_pct], 
                      labels=["Likely", "Unlikely"], 
                      autopct='%1.1f%%',
                      startangle=90,
                      colors=['#66bb6a', '#ff7043'],
                      explode=(0.1, 0),
                      textprops={'fontsize': 12})
            ax[1].set_title("Seasonal Vaccination", pad=20, fontweight='bold')
            
            st.pyplot(fig)
        
        with tab2:
            # Plotly interactive charts
            fig = px.bar(
                x=["H1N1", "Seasonal"],
                y=[h1_vax_pct, seas_vax_pct],
                color=["H1N1", "Seasonal"],
                color_discrete_map={
                    "H1N1": "#81c784",
                    "Seasonal": "#66bb6a"
                },
                labels={'x': 'Vaccine Type', 'y': 'Likelihood (%)'},
                text=[f"{h1_vax_pct:.1f}%", f"{seas_vax_pct:.1f}%"],
                height=400
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                title="Vaccination Likelihood Comparison",
                showlegend=False,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Donut chart
            fig = px.pie(
                names=["H1N1 Likely", "H1N1 Unlikely", "Seasonal Likely", "Seasonal Unlikely"],
                values=[h1_vax_pct, 100-h1_vax_pct, seas_vax_pct, 100-seas_vax_pct],
                hole=0.4,
                color_discrete_sequence=['#81c784', '#ff8a65', '#66bb6a', '#ff7043']
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)