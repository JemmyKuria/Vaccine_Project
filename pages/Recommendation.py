import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional

# Page Configuration
def configure_page():
    st.set_page_config(page_title="AI Recommendation Engine", layout="wide")
    st.title("🤖 AI-Powered Vaccine Recommendations")

# Data Analysis Functions
class VaccineAnalyzer:
    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze dataset and identify key patterns"""
        analysis = {'high_risk_groups': {}, 'behavior_factors': {}, 'medical_factors': {}}
        
        # Prepare data
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        # 1. Identify high-risk groups
        analysis['high_risk_groups'] = VaccineAnalyzer._find_high_risk_groups(df, categorical_cols)
        
        # 2. Find behavioral factors
        behavior_cols = [c for c in df.columns if any(x in c.lower() for x in ['opinion', 'behavior'])]
        analysis['behavior_factors'] = VaccineAnalyzer._analyze_factors(
            df, behavior_cols, correlation_threshold=0.2
        )
        
        # 3. Find medical predictors
        medical_cols = [c for c in df.columns if any(x in c.lower() for x in ['doctor', 'health'])]
        analysis['medical_factors'] = VaccineAnalyzer._analyze_medical_factors(df, medical_cols)
        
        return analysis

    @staticmethod
    def _find_high_risk_groups(df: pd.DataFrame, categorical_cols: List[str]) -> Dict:
        """Identify groups with low vaccination rates"""
        high_risk = {}
        # First drop the specified columns from categorical_cols
        columns_to_drop = ["employment_industry", "employment_occupation", 
                      "hhs_geo_region", "census_msa"]
        filtered_cols = [col for col in categorical_cols if col not in columns_to_drop]
        for col in categorical_cols:
            if df[col].nunique() >= 20:
                continue
                
            group_stats = df.groupby(col)[['h1n1_label', 'seasonal_label']].mean()
            high_risk_groups = group_stats[group_stats.mean(axis=1) < 0.4]  # Threshold
            
            if not high_risk_groups.empty:
                high_risk[col] = high_risk_groups.mean(axis=1).sort_values().to_dict()
        return high_risk

    @staticmethod
    def _analyze_factors(df: pd.DataFrame, cols: List[str], correlation_threshold: float = 0.2) -> Dict:
        """Calculate correlation between factors and vaccination labels"""
        factors = {}
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            corr = df[['h1n1_label', 'seasonal_label']].corrwith(df[col]).mean()
            if abs(corr) > correlation_threshold:
                factors[col] = {
                    'correlation': float(corr),
                    'direction': 'Negative' if corr < 0 else 'Positive'
                }
        return factors

    @staticmethod
    def _analyze_medical_factors(df: pd.DataFrame, cols: List[str]) -> Dict:
        """Analyze impact of medical factors"""
        factors = {}
        for col in cols:
            if df[col].nunique() >= 5:
                continue
                
            effect_size = df.groupby(col)['h1n1_label'].mean().diff().abs().max()
            if not pd.isna(effect_size) and effect_size > 0.15:
                factors[col] = float(effect_size)
        return factors

# Recommendation Generator
class RecommendationEngine:
    @staticmethod
    def generate_recommendations(analysis: Dict) -> Dict[str, Dict]:
        """Convert analysis into actionable recommendations"""
        return {
            "Target Groups": RecommendationEngine._generate_group_recommendations(analysis),
            "Behavioral Factors": RecommendationEngine._generate_behavior_recommendations(analysis),
            "Medical Factors": RecommendationEngine._generate_medical_recommendations(analysis)
        }

    @staticmethod
    def _generate_group_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for col, groups in analysis.get('high_risk_groups', {}).items():
            if not isinstance(groups, dict):
                continue
                
            for group_name, risk_score in groups.items():
                try:
                    risk_score = float(risk_score)
                    key = f"{col.replace('_', ' ')}: {group_name}"
                    recommendations[key] = {
                        "insight": f"{group_name} have {int((1-risk_score)*100)}% higher hesitancy",
                        "numeric_value": int((1-risk_score)*100),
                        "action": f"Targeted education for {group_name}",
                        "priority": "High" if risk_score < 0.3 else "Medium"
                    }
                except (ValueError, TypeError):
                    continue
        return recommendations

    @staticmethod
    def _generate_behavior_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for factor, stats in analysis.get('behavior_factors', {}).items():
            if not isinstance(stats, dict):
                continue
                
            recommendations[factor.replace('_', ' ')] = {
                "insight": f"Strong {stats['direction']} correlation (r={abs(stats['correlation']):.2f})",
                "numeric_value": abs(stats['correlation']),
                "action": f"Campaign focusing on {factor.replace('_', ' ')}",
                "priority": "High"
            }
        return recommendations

    @staticmethod
    def _generate_medical_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for factor, effect in analysis.get('medical_factors', {}).items():
            try:
                effect = float(effect)
                recommendations[factor.replace('_', ' ')] = {
                    "insight": f"Increases likelihood by {int(effect*100)}% when present",
                    "numeric_value": int(effect*100),
                    "action": f"Healthcare provider engagement about {factor.replace('_', ' ')}",
                    "priority": "Critical" if effect > 0.3 else "High"
                }
            except (ValueError, TypeError):
                continue
        return recommendations

# Dashboard Components
# [Previous imports and configuration remain the same...]

class Dashboard:
    @staticmethod
    def show_priority_groups(recommendations: Dict, df: pd.DataFrame):
        st.header("🚨 Top 10 High-Risk Groups")
        
        # Add vaccine type selector
        vaccine_type = st.radio(
            "Select Vaccine Type:",
            ["H1N1", "Seasonal", "Both"],
            horizontal=True,
            index=2
        )
        
        # Filter data based on selection
        if vaccine_type == "H1N1":
            df = df[df['h1n1_label'] == 0]
            risk_col = 'h1n1_label'
        elif vaccine_type == "Seasonal":
            df = df[df['seasonal_label'] == 0]
            risk_col = 'seasonal_label'
        else:  # Both
            df = df[(df['h1n1_label'] == 0) | (df['seasonal_label'] == 0)]
            risk_col = ['h1n1_label', 'seasonal_label']
        
        # Get top 10 high-risk groups
        target_groups = recommendations.get("Target Groups", {})
        sorted_groups = sorted(
            target_groups.items(),
            key=lambda x: x[1].get('numeric_value', 0),
            reverse=True
        )[:10]
        
        # Display as cards
        cols = st.columns(2)
        for i, (group, details) in enumerate(sorted_groups):
            with cols[i % 2]:
                with st.container(border=True):
                    # Header with risk level color
                    priority = details.get('priority', 'Medium')
                    color = "#ff4b4b" if priority == "High" else "#ffa44b"
                    st.markdown(
                        f"<h4 style='color:{color}'>"
                        f"🔴 {group} ({priority} Priority)"
                        f"</h4>", 
                        unsafe_allow_html=True
                    )
                    
                    # Main metrics
                    st.metric(
                        "Hesitancy Rate", 
                        f"{details.get('numeric_value', 0)}%",
                        help="Percentage more likely to refuse vaccination than average"
                    )
                    
                    # Insights and actions
                    st.markdown(f"**Why**: {details.get('insight', '')}")
                    st.markdown(f"**Action**: {details.get('action', '')}")
                    
                    # Show sample size if available
                    group_name = group.split(": ")[-1]
                    col_name = group.split(":")[0].strip()
                    if col_name in df.columns:
                        sample_size = df[df[col_name] == group_name].shape[0]
                        st.caption(f"📊 {sample_size} people in this group")



    @staticmethod
    def show_factors(recommendations: Dict):
        st.header("Top Predictive Features")
        
        # Create a combined dataframe of all factors
        factors_data = []
        
        # Process Behavioral Factors
        for factor, details in recommendations.get("Behavioral Factors", {}).items():
            factors_data.append({
                "Feature": factor,
                "Type": "Behavioral",
                "Impact Score": details.get('numeric_value', 0),
                "Direction": details.get('insight', '').split()[1],  # Gets "Positive" or "Negative"
                "Action": details.get('action', '')
            })
        
        # Process Medical Factors
        for factor, details in recommendations.get("Medical Factors", {}).items():
            factors_data.append({
                "Feature": factor,
                "Type": "Medical",
                "Impact Score": details.get('numeric_value', 0),
                "Direction": "Positive",  # Medical factors are always positive impact
                "Action": details.get('action', '')
            })
        
        # Create and display the table
        if factors_data:
            df = pd.DataFrame(factors_data)
            # Sort by Impact Score (descending)
            df = df.sort_values("Impact Score", ascending=False)
            
            # Display as a table with some styling
            st.dataframe(
                df.style
                .background_gradient(subset=["Impact Score"], cmap="YlOrRd")
                .format({"Impact Score": "{:.2f}"}),
                use_container_width=True
            )
        else:
            st.warning("No significant factors found")

    @staticmethod
    def show_analysis_report(analysis: Dict, recommendations: Dict):
        st.header("Complete Analysis Report")
        
        # Risk Distribution Pie Chart
        risk_counts = {'High': 0, 'Medium': 0}
        for group in recommendations.get("Target Groups", {}).values():
            priority = group.get('priority', 'Medium')
            if priority in risk_counts:
                risk_counts[priority] += 1
        
        if sum(risk_counts.values()) > 0:
            fig1 = px.pie(
                names=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                title="Risk Level Distribution",
                color=list(risk_counts.keys()),
                color_discrete_map={'High': 'red', 'Medium': 'orange'}
            )
            st.plotly_chart(fig1, use_container_width=True)

    @staticmethod
    def setup_export(analysis: Dict, recommendations: Dict):
        st.sidebar.header("Export Options")
        
        # Prepare clean export data
        export_data = {
            "analysis": analysis,
            "recommendations": recommendations,
            "top_groups": [
                {"group": k, **v} 
                for k, v in recommendations.get("Target Groups", {}).items()
            ][:10],  # Only top 10 groups
            "top_factors": [
                {"feature": k, **v} 
                for k, v in {
                    **recommendations.get("Behavioral Factors", {}),
                    **recommendations.get("Medical Factors", {})
                }.items()
            ]
        }
        
        # JSON Export
        st.sidebar.download_button(
            "📥 Download Full Report (JSON)",
            data=pd.json_normalize(export_data).to_json(orient='records'),
            file_name="vaccine_recommendations.json"
        )
        
        # CSV Export (simplified)
        csv_data = []
        for group in export_data["top_groups"]:
            csv_data.append({
                "Type": "Group",
                "Name": group.get("group", ""),
                "Score": group.get("numeric_value", 0),
                "Priority": group.get("priority", ""),
                "Action": group.get("action", "")
            })
        
        for factor in export_data["top_factors"]:
            csv_data.append({
                "Type": "Factor",
                "Name": factor.get("feature", ""),
                "Score": factor.get("numeric_value", 0),
                "Priority": factor.get("priority", ""),
                "Action": factor.get("action", "")
            })
        
        st.sidebar.download_button(
            "📊 Executive Summary (CSV)",
            data=pd.DataFrame(csv_data).to_csv(index=False),
            file_name="vaccine_recommendations.csv"
        )

# [Rest of the code remains the same...]

# Main Application
def main():
    configure_page()
    
    # Check for data
    if "results_df" not in st.session_state:
        st.warning("Please process data on the Home page first.")
        st.stop()
    
    # Analyze data
    analysis = VaccineAnalyzer.analyze_data(st.session_state["results_df"])
    recommendations = RecommendationEngine.generate_recommendations(analysis)
    
    # Setup tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Priority Groups", "🧠 Top Factors", "🔄 Analysis", "📤 Export"])
    
    with tab1:
        Dashboard.show_priority_groups(recommendations, st.session_state["results_df"])    
    with tab2:
        Dashboard.show_factors(recommendations)
    
    with tab3:
        Dashboard.show_analysis_report(analysis, recommendations)
    
    with tab4:
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()