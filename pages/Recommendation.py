import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional
from faker import Faker

# Initialize Faker
fake = Faker()

# Page Configuration
def configure_page():
    st.set_page_config(page_title="AI Recommendation Engine", layout="wide")
    st.title("🤖 AI-Powered Vaccine Recommendations")

# Data Analysis Functions
class VaccineAnalyzer:
    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze dataset and identify key patterns"""
        analysis = {
            'high_risk_groups': {},
            'behavior_factors': {},
            'medical_factors': {},
            'barrier_profiles': {}
        }
        
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
        
        # 4. Add barrier analysis
        analysis['barrier_profiles'] = VaccineAnalyzer._analyze_barriers(df)
        
        return analysis

    @staticmethod
    def _find_high_risk_groups(df: pd.DataFrame, categorical_cols: List[str]) -> Dict:
        """Identify groups with low vaccination rates"""
        high_risk = {}
        columns_to_drop = ["employment_industry", "employment_occupation", 
                          "hhs_geo_region", "census_msa"]
        filtered_cols = [col for col in categorical_cols if col not in columns_to_drop]
        
        for col in filtered_cols:
            if df[col].nunique() >= 20:
                continue
                
            group_stats = df.groupby(col)[['h1n1_label', 'seasonal_label']].mean()
            high_risk_groups = group_stats[group_stats.mean(axis=1) < 0.4]
            
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

    @staticmethod
    def _analyze_barriers(df: pd.DataFrame) -> Dict:
        """Analyze vaccine hesitancy barriers and generate messages"""
        barrier_messages = {
            'Cost / No insurance': {
                'h1n1': "Hi {name}, the H1N1 flu vaccine is completely free for everyone. No insurance is needed — protect yourself today!",
                'seasonal': "Hi {name}, the seasonal flu shot is free for everyone. No insurance is required — stay healthy this flu season!"
            },
            'Perceived low risk': {
                'h1n1': "Hi {name}, even healthy people can catch H1N1. The virus spreads easily — protect yourself and your loved ones by getting vaccinated.",
                'seasonal': "Hi {name}, seasonal flu infects millions each year, even healthy people. Get your free shot to stay safe."
            },
            'Safety concerns': {
                'h1n1': "Hi {name}, the H1N1 vaccine has been thoroughly tested for safety. It's a safe way to protect yourself from serious illness.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is safe and prevents thousands of hospital visits each year. Get protected today."
            },
            'Lack of time / access': {
                'h1n1': "Hi {name}, getting the H1N1 shot takes less than 10 minutes and is available at your nearest clinic. No appointment needed.",
                'seasonal': "Hi {name}, the seasonal flu shot is quick and available near you. Walk in today and protect yourself in minutes."
            },
            'Misinformation': {
                'h1n1': "Hi {name}, H1N1 is not the same as a cold — it can cause serious complications. Vaccination is the best protection.",
                'seasonal': "Hi {name}, flu is more dangerous than a cold. The seasonal flu vaccine is your best defense."
            }
        }

        # Detect barriers based on your specified columns
        df['barrier'] = df.apply(lambda row: 
            'Cost / No insurance' if row['health_insurance'] == 0 else
            'Perceived low risk' if row['opinion_h1n1_risk'] <= 2 else
            'Safety concerns' if row['opinion_h1n1_vacc_effective'] <= 2 else
            'Lack of time / access' if row['behavioral_antiviral_meds'] == 0 else
            'Misinformation', axis=1)

        # Generate messages
        messages = []
        for _, row in df.iterrows():
            messages.append({
                'barrier': row['barrier'],
                'h1n1_message': barrier_messages[row['barrier']]['h1n1'].format(name=fake.first_name()),
                'seasonal_message': barrier_messages[row['barrier']]['seasonal'].format(name=fake.first_name()),
                'priority': 'High' if row['barrier'] in ['Cost / No insurance', 'Safety concerns'] else 'Medium'
            })
        
        return messages

# Recommendation Generator
class RecommendationEngine:
    @staticmethod
    def generate_recommendations(analysis: Dict) -> Dict[str, Dict]:
        """Convert analysis into actionable recommendations"""
        return {
            "Target Groups": RecommendationEngine._generate_group_recommendations(analysis),
            "Behavioral Factors": RecommendationEngine._generate_behavior_recommendations(analysis),
            "Medical Factors": RecommendationEngine._generate_medical_recommendations(analysis),
            "Barrier Messages": RecommendationEngine._generate_barrier_recommendations(analysis)
        }

    @staticmethod
    def _generate_group_recommendations(analysis: Dict) -> Dict:
        """Generate recommendations for high-risk groups"""
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
        """Generate recommendations for behavioral factors"""
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
        """Generate recommendations for medical factors"""
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

    @staticmethod
    def _generate_barrier_recommendations(analysis: Dict) -> Dict:
        """Generate recommendations based on barrier profiles"""
        recommendations = {}
        barrier_messages = analysis.get('barrier_profiles', [])
        
        for i, msg in enumerate(barrier_messages[:100]):  # Limit to first 100
            recommendations[f"Barrier {i+1}"] = {
                "insight": f"Detected barrier: {msg['barrier']}",
                "numeric_value": 0,
                "action": f"H1N1: {msg['h1n1_message']}\n\nSeasonal: {msg['seasonal_message']}",
                "priority": msg['priority']
            }
            
        return recommendations

# Dashboard Components
class Dashboard:
    @staticmethod
    def show_priority_groups(recommendations: Dict):
        st.header("Priority Intervention Groups (Sorted by Risk)")
        
        # Get groups and sort by numeric_value (descending)
        target_groups = recommendations.get("Target Groups", {})
        sorted_groups = sorted(
            target_groups.items(),
            key=lambda x: x[1].get('numeric_value', 0),
            reverse=True
        )
        
        # Show top 10 highest risk groups
        cols = st.columns(2)
        for i, (group, details) in enumerate(sorted_groups[:10]):
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

    @staticmethod
    def show_factors(recommendations: Dict):
        st.header("Most Influential Factors")
        
        # Behavioral Factors
        st.subheader("Behavioral Drivers", divider="blue")
        for factor, details in recommendations.get("Behavioral Factors", {}).items():
            with st.container(border=True):
                st.markdown(f"**{factor.title()}**")
                st.progress(min(1.0, details.get('numeric_value', 0)))
                st.caption(details.get('insight', ''))
                st.info(f"💡 **Action**: {details.get('action', '')}")
        
        # Medical Factors
        st.subheader("Healthcare Leverage Points", divider="green")
        for factor, details in recommendations.get("Medical Factors", {}).items():
            with st.container(border=True):
                cols = st.columns([3,1])
                cols[0].markdown(f"**{factor.title()}**")
                cols[0].caption(details.get('insight', ''))
                cols[0].info(f"💡 **Action**: {details.get('action', '')}")
                cols[1].metric("Impact", f"+{details.get('numeric_value', 0)}%")

    @staticmethod
    def show_barrier_messages(recommendations: Dict):
        st.header("📨 Personalized Messaging Recommendations")
        
        barrier_recs = recommendations.get("Barrier Messages", {})
        if not barrier_recs:
            st.warning("No barrier messages generated")
            return
        
        # Vaccine type selector
        vaccine_type = st.radio(
            "Select Vaccine Type:",
            ["H1N1", "Seasonal", "Both"],
            horizontal=True,
            index=2
        )
        
        # Show messages
        st.subheader(f"Top {min(5, len(barrier_recs))} Message Templates")
        for i, (_, details) in enumerate(list(barrier_recs.items())[:5]):
            with st.expander(f"Template {i+1} - {details['priority']} Priority"):
                if vaccine_type in ["H1N1", "Both"]:
                    st.info(f"**H1N1 Message**: {details['action'].split('H1N1: ')[1].split('\n\n')[0]}")
                if vaccine_type in ["Seasonal", "Both"]:
                    st.success(f"**Seasonal Message**: {details['action'].split('Seasonal: ')[1]}")
                
                st.caption(f"**For**: {details['insight'].replace('Detected barrier: ', '')}")

    @staticmethod
    def show_analysis_report(analysis: Dict, recommendations: Dict):
        st.header("Complete Analysis Report")
        
        # Feature Importance Visualization
        features = []
        importance = []
        
        for factor_type in ["Behavioral Factors", "Medical Factors"]:
            for details in recommendations.get(factor_type, {}).values():
                if 'r=' in details.get('insight', ''):
                    importance.append(abs(float(details['insight'].split('r=')[1][:4])))
                else:
                    importance.append(details.get('numeric_value', 0) / 100)
                features.append(details.get('action', '').split('about ')[-1].title())
        
        if features and importance:
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                color=importance,
                color_continuous_scale='Teal',
                labels={'x': 'Impact Score', 'y': ''},
                title="Feature Impact Ranking"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw Analysis Data
        with st.expander("📊 View Analysis Details"):
            st.json(analysis)

    @staticmethod
    def setup_export(analysis: Dict, recommendations: Dict):
        st.sidebar.header("Export Options")
        
        # Prepare export data
        export_data = []
        for category, items in recommendations.items():
            for name, details in items.items():
                export_data.append({
                    "Type": category,
                    "Name": name,
                    "Insight": details.get('insight', ''),
                    "Action": details.get('action', ''),
                    "Priority": details.get('priority', ''),
                    "Score": details.get('numeric_value', 0)
                })
        
        df = pd.DataFrame(export_data)
        
        # JSON Export
        st.sidebar.download_button(
            "📥 Download Full Report (JSON)",
            data=df.to_json(orient='records'),
            file_name="vaccine_recommendations.json"
        )
        
        # CSV Export
        st.sidebar.download_button(
            "📊 Executive Summary (CSV)",
            data=df.to_csv(index=False),
            file_name="vaccine_recommendations.csv"
        )

# Main Application
def main():
    configure_page()
    
    # Check for data
    if "results_df" not in st.session_state:
        st.warning("Please process data on the Home page first.")
        st.stop()
    
    # Analyze data
    with st.spinner("Analyzing vaccination data..."):
        analysis = VaccineAnalyzer.analyze_data(st.session_state["results_df"])
        recommendations = RecommendationEngine.generate_recommendations(analysis)
    
    # Setup tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Priority Groups", 
        "🧠 Top Factors", 
        "📨 Messaging",
        "📈 Analysis", 
        "📤 Export"
    ])
    
    with tab1:
        Dashboard.show_priority_groups(recommendations)
    
    with tab2:
        Dashboard.show_factors(recommendations)
    
    with tab3:
        Dashboard.show_barrier_messages(recommendations)
    
    with tab4:
        Dashboard.show_analysis_report(analysis, recommendations)
    
    with tab5:
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()