import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Vaccine Dashboard", layout="wide")
st.title("📊 Advanced Vaccination Prediction Dashboard")

# 1. Data Loading
if "results_df" not in st.session_state:
    st.warning("Please process data on the Home page first.")
    st.stop()

results = st.session_state["results_df"]

# 2. Key Metrics Dashboard
st.header("📈 Key Metrics")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total Respondents", f"{len(results):,}")
metric2.metric("H1N1 Non-Takers", f"{(results['h1n1_label'] == 0).sum():,}", 
               help="Predicted not to get H1N1 vaccine")
metric3.metric("Seasonal Non-Takers", f"{(results['seasonal_label'] == 0).sum():,}",
               help="Predicted not to get seasonal vaccine")
metric4.metric("Double Non-Takers", 
               f"{((results['h1n1_label'] == 0) & (results['seasonal_label'] == 0)).sum():,}",
               help="Predicted to take neither vaccine")

# 3. Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "🧠 Behavioral Insights", "🔄 Relationships", "📤 Export"])

with tab1:  # Demographic Analysis
    st.subheader("Population Breakdown")
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        demo_factor = st.selectbox(
            "Select demographic factor:",
            options=results.select_dtypes(include=['object', 'category']).columns,
            index=0
        )
    
    with demo_col2:
        view_type = st.radio("View:", ["Count", "Percentage"], horizontal=True)
    
    # Demographic bar chart
    demo_data = results.groupby(demo_factor)[['h1n1_label', 'seasonal_label']].mean().reset_index()
    if view_type == "Count":
        demo_data = results.groupby(demo_factor)[['h1n1_label', 'seasonal_label']]\
                          .apply(lambda x: (x == 0).sum()).reset_index()
    
    fig = px.bar(
        demo_data,
        x=demo_factor,
        y=['h1n1_label', 'seasonal_label'],
        barmode='group',
        title=f"Vaccine Hesitancy by {demo_factor.replace('_', ' ').title()}",
        labels={'value': 'Count' if view_type == "Count" else 'Percentage', 'variable': 'Vaccine'},
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Demographic pie charts
    st.subheader("Population Composition")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(results, names=demo_factor, title=f"Distribution by {demo_factor}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        selected_vaccine = st.selectbox("Show uptake for:", ['h1n1_label', 'seasonal_label'])
        fig = px.pie(results, names=selected_vaccine, title=f"{selected_vaccine.replace('_', ' ').title()} Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab2:  # Behavioral Insights (Simplified)
    st.subheader("Public Perception Analysis")
    
    # Simplified Risk Perception Bars
    st.markdown("### How Perceptions Affect Vaccine Decisions")
    risk_cols = [c for c in results.columns if 'opinion' in c.lower() or 'concern' in c.lower()]
    
    if risk_cols:
        selected_perception = st.selectbox(
            "Select public perception factor:",
            options=risk_cols,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="See how different opinions correlate with vaccination likelihood"
        )
        
        # Simplified bar chart (average scores)
        perception_data = results.groupby(['h1n1_label', 'seasonal_label'])[selected_perception].mean().reset_index()
        perception_data['Vaccine Decision'] = perception_data.apply(
            lambda row: f"H1N1: {'Yes' if row['h1n1_label'] else 'No'}, Seasonal: {'Yes' if row['seasonal_label'] else 'No'}",
            axis=1
        )
        
        fig = px.bar(
            perception_data,
            x='Vaccine Decision',
            y=selected_perception,
            color='Vaccine Decision',
            title=f"Average {selected_perception.replace('_', ' ').title()} Score by Vaccine Decision",
            labels={selected_perception: "Average Score (1-5)"},
            color_discrete_sequence=['#4C78A8', '#54A24B', '#E45756', '#F58518']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clear Text Interpretation
        st.markdown("#### What This Means")
        low_score = perception_data[selected_perception].min()
        high_score = perception_data[selected_perception].max()
        
        st.info(f"""
        - People who took **both vaccines** had an average {selected_perception.replace('_', ' ')} score of {perception_data.loc[3, selected_perception]:.1f}/5
        - Those who took **neither vaccine** averaged {perception_data.loc[0, selected_perception]:.1f}/5
        - Higher scores indicate {'more concern' if 'concern' in selected_perception else 'stronger agreement'}
        """)
        
        # Simple Recommendation
        st.markdown("#### Suggested Action")
        if 'safety' in selected_perception:
            st.success("""
            **Focus on safety education:**  
            Address misconceptions through doctor-patient conversations and clear infographics.
            """)
        elif 'risk' in selected_perception:
            st.success("""
            **Highlight personal risk factors:**  
            Tailored messaging showing individual susceptibility may be effective.
            """)

            
with tab3:  # Relationships (Simplified)
    st.subheader("Feature Relationships")
    
    # Simplified Correlation Heatmap
    num_cols = results.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = results[num_cols].corr()
    
    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Simplified Parallel Coordinates
    st.subheader("Multivariate Analysis")
    parallel_vars = st.multiselect(
        "Select variables for parallel coordinates:",
        num_cols,
        default=num_cols[:4] if len(num_cols) > 3 else num_cols
    )
    
    if len(parallel_vars) > 1:
        fig = px.parallel_coordinates(
            results,
            dimensions=parallel_vars,
            color='h1n1_label',
            color_continuous_scale=['#EF553B', '#636EFA']
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:  # Data Export
    st.subheader("Export Results")
    
    with st.expander("📄 Preview Data"):
        st.dataframe(results.head(100))
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Full Data (CSV)",
            data=results.to_csv(index=False),
            file_name="vaccine_predictions_full.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download Filtered Data (CSV)",
            data=results[['respondent_id', 'h1n1_label', 'seasonal_label']].to_csv(index=False),
            file_name="vaccine_predictions_minimal.csv",
            mime="text/csv"
        )
    
    st.subheader("Save Visualizations")
    viz_option = st.selectbox("Select visualization to export:", ["Demographic Breakdown", "Risk Factors", "Correlation Matrix"])
    if viz_option:
        st.warning("Visualization export requires Plotly's static image export. Install kaleido with: pip install kaleido")