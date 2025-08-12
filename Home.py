import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pipeline import preprocess, predict 
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Vaccine Campaign Optimizer", 
    page_icon="💉",
    layout="wide"
)

# Load header image (replace with your own image path)
try:
    header_img = Image.open("vaccine_campaign.jpg")  # Place your image in same directory
    st.image(header_img, use_column_width=True, caption="Optimizing Vaccine Outreach")
except:
    st.warning("Header image not found - using placeholder")
    # Fallback colored header
    st.markdown("""
    <div style='background-color:#2e7d32; padding:20px; border-radius:10px; margin-bottom:20px'>
        <h1 style='color:white; text-align:center'>Vaccine Campaign Optimizer</h1>
    </div>
    """, unsafe_allow_html=True)

# About Section with columns
st.header("📊 About This Tool")
about_col1, about_col2 = st.columns([3, 1])

with about_col1:
    st.markdown("""
    This application helps **public health officials** and **vaccine campaign managers**:
    
    - 🔍 Identify low-vaccination populations  
    - 📈 Analyze behavioral/demographic factors  
    - ✉️ Recommend targeted messaging strategies  
    """)
    
    st.markdown("""
    ### How It Works
    1. **Upload** survey data (CSV format)  
    2. **Analyze** vaccination likelihood predictions  
    3. **Explore** targeted campaign strategies  
    """)

with about_col2:
    # Placeholder for an icon or simple visualization
    st.markdown("""
    <div style='text-align:center'>
        <span style='font-size:80px'>💉</span>
        <p><em>Maximizing vaccine campaign effectiveness</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# File Upload Section with visual feedback
st.header("📂 1. Upload Survey Data")
uploaded = st.file_uploader("Choose your survey data file", type=["csv"],
                           help="Upload your CSV file containing survey responses")

if uploaded is None:
    st.info("👋 Please upload your survey data to begin analysis")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"✅ Successfully loaded {len(df_raw)} survey responses")

# Data preview with expandable section
with st.expander("🔍 View Sample Data", expanded=False):
    st.dataframe(df_raw.head().style.highlight_max(axis=0, color='#c8e6c9'))

st.markdown("---")

# Analysis Section with animated elements
st.header("🔬 2. Generate Predictions")

if st.button("🚀 Analyze Vaccination Likelihood", type="primary", 
            help="Process the data and generate predictions"):
    
    with st.spinner("Crunching numbers... This may take a moment"):
        # Clean and predict
        df_clean = preprocess(df_raw.copy())
        h1n1_label, seasonal_label = predict(df_clean)
        
        # Store results
        results = df_raw.copy()
        results["h1n1_label"] = h1n1_label
        results["seasonal_label"] = seasonal_label
        st.session_state["results_df"] = results
        
        # Success animation
        st.balloons()
        st.success("✨ Analysis complete! Visit the Recommendations page for campaign strategies.")
        
        # Metrics with visual cards
        total = len(results)
        h1_vax_pct = results["h1n1_label"].mean() * 100
        seas_vax_pct = results["seasonal_label"].mean() * 100
        
        st.subheader("📊 Results Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Respondents", f"{total:,}", 
                    help="Total number of survey responses analyzed")
        
        with col2:
            st.metric("H1N1 Vaccine Likely", f"{h1_vax_pct:.1f}%", 
                    delta=f"{h1_vax_pct-50:.1f}% from baseline" if h1_vax_pct else None,
                    delta_color="normal",
                    help="Percentage likely to get H1N1 vaccine")
        
        with col3:
            st.metric("Seasonal Vaccine Likely", f"{seas_vax_pct:.1f}%", 
                    delta=f"{seas_vax_pct-50:.1f}% from baseline" if seas_vax_pct else None,
                    delta_color="normal",
                    help="Percentage likely to get seasonal flu vaccine")

        # Enhanced visualization section
        st.subheader("📈 Vaccination Likelihood Distribution")
        
        tab1, tab2 = st.tabs(["Pie Charts", "Interactive View"])
        
        with tab1:
            # Matplotlib pie charts
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # H1N1 pie chart
            ax[0].pie([h1_vax_pct, 100-h1_vax_pct], 
                     labels=["Likely", "Unlikely"], 
                     autopct='%1.1f%%',
                     startangle=90,
                     colors=['#4CAF50', '#f44336'],
                     explode=(0.1, 0),
                     textprops={'fontsize': 10})
            ax[0].set_title("H1N1 Vaccination", fontweight='bold')
            
            # Seasonal pie chart
            ax[1].pie([seas_vax_pct, 100-seas_vax_pct], 
                      labels=["Likely", "Unlikely"], 
                      autopct='%1.1f%%',
                      startangle=90,
                      colors=['#4CAF50', '#f44336'],
                      explode=(0.1, 0),
                      textprops={'fontsize': 10})
            ax[1].set_title("Seasonal Vaccination", fontweight='bold')
            
            st.pyplot(fig)
        
        with tab2:
            # Plotly interactive chart
            fig = px.bar(
                x=["H1N1 Vaccine", "Seasonal Vaccine"],
                y=[h1_vax_pct, seas_vax_pct],
                color=["H1N1 Vaccine", "Seasonal Vaccine"],
                color_discrete_sequence=["#4CAF50", "#2e7d32"],
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

        # Recommendation prompt
        st.markdown("---")
        st.markdown("""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px'>
            <h3>Ready for Campaign Recommendations?</h3>
            <p>Visit the <strong>Recommendations</strong> page to see targeted strategies based on this analysis.</p>
        </div>
        """, unsafe_allow_html=True)