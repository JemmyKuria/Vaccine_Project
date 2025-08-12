import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional
from twilio.rest import Client
import plotly.graph_objects as go

# Access Twilio secrets
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
from_number = st.secrets["twilio"]["from_number"]

# Initialize Twilio client
client = Client(account_sid, auth_token)

# ---------------- Page Configuration ----------------
def configure_page():
    st.set_page_config(page_title="AI Vaccine Recommendation Engine", layout="wide")
    st.title("🤖 AI-Powered Vaccine Recommendations")
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-b7 {
        color: white;
    }
    .st-c0 {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Analysis & Barrier Logic ----------------
class VaccineAnalyzer:
    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze dataset and identify key patterns with robust error handling"""
        analysis = {
            'high_risk_groups': {},
            'behavior_factors': {},
            'medical_factors': {},
            'barrier_profiles': [],
            'dataset_stats': {}
        }

        try:
            # Dataset statistics
            analysis['dataset_stats'] = {
                'total_records': len(df),
                'vaccination_rate_h1n1': df.get('h1n1_label', pd.Series(0)).mean(),
                'vaccination_rate_seasonal': df.get('seasonal_label', pd.Series(0)).mean()
            }

            # Clean categorical data
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                df[col] = df[col].astype(str).replace('nan', 'Unknown')

            # 1. Identify high-risk groups (excluding NaN groups)
            analysis['high_risk_groups'] = VaccineAnalyzer._find_high_risk_groups(df, categorical_cols)

            # 2. Behavior factors with enhanced correlation handling
            behavior_cols = [c for c in df.columns if any(x in c.lower() for x in ['opinion', 'behavior'])]
            analysis['behavior_factors'] = VaccineAnalyzer._analyze_factors(df, behavior_cols)

            # 3. Medical factors with improved effect size calculation
            medical_cols = [c for c in df.columns if any(x in c.lower() for x in ['doctor', 'health'])]
            analysis['medical_factors'] = VaccineAnalyzer._analyze_medical_factors(df, medical_cols)

            # 4. Barriers with strict type checking
            analysis['barrier_profiles'] = VaccineAnalyzer._analyze_barriers(df)

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {}

        return analysis

    @staticmethod
    def _find_high_risk_groups(df: pd.DataFrame, categorical_cols: List[str]) -> Dict:
        """Identify groups with low vaccination rates, excluding NaN/Unknown groups"""
        high_risk = {}
        exclude_cols = ["employment_industry", "employment_occupation", "hhs_geo_region", 
                       "census_msa", "name", "phone_number", "respondent_id", 
                       "h1n1_vaccine", "seasonal_vaccine", "h1n1_vaccine_pred", 
                       "seasonal_vaccine_pred", "h1n1_label", "seasonal_label"]

        for col in [c for c in categorical_cols if c not in exclude_cols and df[c].nunique() < 20]:
            try:
                group_stats = df[df[col] != 'Unknown'].groupby(col)[['h1n1_label', 'seasonal_label']].mean()
                high_risk_groups = group_stats[group_stats.mean(axis=1) < 0.4]
                
                if not high_risk_groups.empty:
                    high_risk[col] = {
                        group: float(rate) 
                        for group, rate in high_risk_groups.mean(axis=1).sort_values().items()
                        if group != 'Unknown' and not pd.isna(rate)
                    }
            except Exception:
                continue
                
        return {k: v for k, v in high_risk.items() if v}  # Remove empty entries

    @staticmethod
    def _analyze_factors(df: pd.DataFrame, cols: List[str]) -> Dict:
        """Calculate correlation between factors and vaccination labels with robust handling"""
        factors = {}
        required_cols = {'h1n1_label', 'seasonal_label'}
        
        if not required_cols.issubset(df.columns):
            return factors

        for col in [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]:
            try:
                h1n1_corr = df['h1n1_label'].corr(df[col])
                seasonal_corr = df['seasonal_label'].corr(df[col])
                avg_corr = (h1n1_corr + seasonal_corr) / 2
                
                if abs(avg_corr) > 0.15 and not pd.isna(avg_corr):
                    factors[col] = {
                        'correlation': float(avg_corr),
                        'direction': 'Negative' if avg_corr < 0 else 'Positive',
                        'h1n1_corr': float(h1n1_corr) if not pd.isna(h1n1_corr) else 0,
                        'seasonal_corr': float(seasonal_corr) if not pd.isna(seasonal_corr) else 0
                    }
            except Exception:
                continue
                
        return factors

    @staticmethod
    def _analyze_medical_factors(df: pd.DataFrame, cols: List[str]) -> Dict:
        """Analyze impact of medical factors with improved effect size calculation"""
        factors = {}
        if 'h1n1_label' not in df.columns:
            return factors

        for col in [c for c in cols if df[c].nunique() < 5]:
            try:
                group_means = df.groupby(col)['h1n1_label'].mean()
                if len(group_means) >= 2:
                    effect_size = group_means.max() - group_means.min()
                    if effect_size > 0.1:
                        factors[col] = {
                            'effect_size': float(effect_size),
                            'high_group': group_means.idxmax(),
                            'low_group': group_means.idxmin()
                        }
            except Exception:
                continue
                
        return factors

    @staticmethod
    def _analyze_barriers(df: pd.DataFrame) -> List[Dict]:
        """Analyze barriers with strict type checking and complete seasonal/H1N1 separation"""
        try:
            df = df.copy()
            
            # Target population - those predicted/actual not vaccinated
            if all(c in df.columns for c in ['h1n1_vaccine_pred', 'seasonal_vaccine_pred']):
                df_target = df[(df['h1n1_vaccine_pred'] == 0) | (df['seasonal_vaccine_pred'] == 0)].copy()
            elif 'h1n1_label' in df.columns:
                df_target = df[df['h1n1_label'] == 0].copy()
            else:
                df_target = df.copy()

            # Barrier definitions with enhanced seasonal/H1N1 separation
            barrier_conditions = [
                ('No Insurance', 
                 (df_target.get('health_insurance', 1) == 0)),
                ('Low Vaccine Belief', 
                 (df_target.get('opinion_h1n1_vacc_effective', 3) <= 2) |
                 (df_target.get('opinion_seasonal_vacc_effective', 3) <= 2)),
                ('Low Risk Perception', 
                 (df_target.get('opinion_h1n1_risk', 3) <= 2) |
                 (df_target.get('opinion_seasonal_risk', 3) <= 2)),
                ('Low Knowledge', 
                 (df_target.get('h1n1_knowledge', 2) <= 1) |
                 (df_target.get('seasonal_knowledge', 2) <= 1)),
                ('Access Issues', 
                 (df_target.get('behavioral_antiviral_meds', 0) == 0)),
                ('Low Safe Behaviors', 
                 (df_target.get('safe_behavior_score', 10) <= 2))
            ]

            # Assign each person to exactly one barrier
            df_target['barrier_profile'] = 'No Major Barrier'
            for barrier, condition in barrier_conditions:
                mask = condition & (df_target['barrier_profile'] == 'No Major Barrier')
                df_target.loc[mask, 'barrier_profile'] = barrier

            # Enhanced message templates with clear seasonal/H1N1 separation
            barrier_messages = {
                'No Insurance': {
                    'h1n1': "Hi {name}, the H1N1 vaccine is completely free with no insurance required. Protect yourself today at any local clinic.",
                    'seasonal': "Hi {name}, seasonal flu shots are free regardless of insurance status. Get vaccinated at pharmacies or health departments."
                },
                'Low Risk Perception': {
                    'h1n1': "Hi {name}, H1N1 can seriously affect healthy adults. Vaccination reduces hospitalization risk by 70%.",
                    'seasonal': "Hi {name}, seasonal flu causes thousands of hospitalizations yearly. Your shot takes just minutes but protects all season."
                },
                'Low Vaccine Belief': {
                    'h1n1': "Hi {name}, the H1N1 vaccine is rigorously tested and recommended by all major health organizations.",
                    'seasonal': "Hi {name}, seasonal flu vaccines are updated annually and proven to reduce illness severity."
                },
                'Low Knowledge': {
                    'h1n1': "Hi {name}, H1N1 spreads through coughs/sneezes. The vaccine safely teaches your immune system to fight the virus.",
                    'seasonal': "Hi {name}, flu viruses change each year requiring updated vaccines. Protection begins about 2 weeks after vaccination."
                },
                'Access Issues': {
                    'h1n1': "Hi {name}, H1N1 shots are available at walk-in clinics with evening/weekend hours for your convenience.",
                    'seasonal': "Hi {name}, seasonal flu shots are offered at pharmacies, workplaces, and pop-up clinics - no appointment needed."
                },
                'Low Safe Behaviors': {
                    'h1n1': "Hi {name}, while handwashing helps, the H1N1 vaccine provides much stronger protection against this serious virus.",
                    'seasonal': "Hi {name}, combine your healthy habits with a seasonal flu shot for optimal protection this winter."
                },
                'No Major Barrier': {
                    'h1n1': "Hi {name}, help protect our community by getting your H1N1 vaccination today.",
                    'seasonal': "Hi {name}, it's not too late to get your seasonal flu shot and stay protected."
                }
            }

            # Build profiles with complete contact handling
            profiles = []
            for profile, group in df_target.groupby('barrier_profile'):
                contacts = []
                if all(c in group.columns for c in ['name', 'phone_number']):
                    contacts = group[['name', 'phone_number']].dropna().to_dict('records')
                
                sample_contact = contacts[0] if contacts else None
                
                profiles.append({
                    'barrier_profile': profile,
                    'people_affected': len(group),
                    'primary_barrier': profile,
                    'h1n1_message': barrier_messages[profile]['h1n1'],
                    'seasonal_message': barrier_messages[profile]['seasonal'],
                    'sample_contact': sample_contact,
                    'affected_contacts': contacts,
                    'priority': 'High' if profile in ['No Insurance', 'Low Vaccine Belief'] else 'Medium'
                })

            return sorted(profiles, key=lambda x: (-x['people_affected'], x['priority']))
            
        except Exception as e:
            st.error(f"Barrier analysis error: {str(e)}")
            return []

# ---------------- Recommendation Engine ----------------
class RecommendationEngine:
    @staticmethod
    def generate_recommendations(analysis: Dict) -> Dict[str, Dict]:
        """Convert analysis into actionable recommendations with strict validation"""
        try:
            return {
                "Target Groups": RecommendationEngine._generate_group_recommendations(analysis),
                "Behavioral Factors": RecommendationEngine._generate_behavior_recommendations(analysis),
                "Medical Factors": RecommendationEngine._generate_medical_recommendations(analysis),
                "Barrier Messages": RecommendationEngine._generate_barrier_recommendations(analysis)
            }
        except Exception:
            return {}

    @staticmethod
    def _generate_group_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for col, groups in analysis.get('high_risk_groups', {}).items():
            for group_name, risk_score in groups.items():
                try:
                    if pd.isna(group_name) or pd.isna(risk_score):
                        continue
                        
                    key = f"{col.replace('_', ' ').title()}: {group_name}"
                    recommendations[key] = {
                        "insight": f"{int((1-risk_score)*100)}% lower vaccination rate than average",
                        "numeric_value": int((1-risk_score)*100),
                        "action": f"Targeted outreach for {group_name} {col.replace('_', ' ')}",
                        "priority": "High" if risk_score < 0.3 else "Medium",
                        "group": group_name,
                        "feature": col
                    }
                except Exception:
                    continue
                    
        return dict(sorted(recommendations.items(), key=lambda x: -x[1]['numeric_value']))

    @staticmethod
    def _generate_behavior_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for factor, stats in analysis.get('behavior_factors', {}).items():
            try:
                factor_name = factor.replace('_', ' ').title()
                direction = "Negative" if stats.get('correlation', 0) < 0 else "Positive"
                recommendations[factor_name] = {
                    "insight": f"{direction} impact (r={abs(stats['correlation']):.2f}",
                    "numeric_value": abs(stats['correlation']),
                    "action": f"Behavioral intervention targeting {factor_name.lower()}",
                    "priority": "High" if abs(stats['correlation']) > 0.25 else "Medium",
                    "h1n1_corr": stats.get('h1n1_corr', 0),
                    "seasonal_corr": stats.get('seasonal_corr', 0),
                    "direction": direction  
                }
            except Exception:
                continue
                
        return dict(sorted(recommendations.items(), key=lambda x: -x[1]['numeric_value']))

    @staticmethod
    def _generate_medical_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for factor, details in analysis.get('medical_factors', {}).items():
            try:
                factor_name = factor.replace('_', ' ').title()
                recommendations[factor_name] = {
                    "insight": f"{int(details['effect_size']*100)}% higher vaccination when present",
                    "numeric_value": int(details['effect_size']*100),
                    "action": f"Healthcare provider education about {factor_name.lower()}",
                    "priority": "High" if details['effect_size'] > 0.25 else "Medium",
                    "high_group": details['high_group'],
                    "low_group": details['low_group']
                }
            except Exception:
                continue
                
        return dict(sorted(recommendations.items(), key=lambda x: -x[1]['numeric_value']))

    @staticmethod
    def _generate_barrier_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        for i, profile in enumerate(analysis.get('barrier_profiles', [])[:100]):  # Limit for UI
            try:
                key = f"Barrier {i+1}: {profile['barrier_profile']}"
                recommendations[key] = {
                    "insight": f"{profile['people_affected']} people affected by {profile['primary_barrier'].lower()}",
                    "numeric_value": profile['people_affected'],
                    "action": f"H1N1: {profile['h1n1_message']}\n\nSeasonal: {profile['seasonal_message']}",
                    "priority": profile['priority'],
                    "sample_contact": profile.get('sample_contact'),
                    "affected_contacts": profile.get('affected_contacts', []),
                    "barrier_type": profile['primary_barrier']
                }
            except Exception:
                continue
                
        return recommendations

# ---------------- Enhanced Dashboard Components ----------------
class Dashboard:
    @staticmethod
    def show_priority_groups(recommendations: Dict):
        st.header("🎯 High-Priority Intervention Groups")
        st.markdown("Demographic groups with significantly lower vaccination rates")
        
        target_groups = recommendations.get("Target Groups", {})
        if not target_groups:
            st.info("No high-risk demographic groups identified")
            return
            
        cols = st.columns(2)
        for i, (group, details) in enumerate(target_groups.items()):
            with cols[i % 2]:
                color = "#ef476f" if details['priority'] == "High" else "#ffd166"
                with st.container():
                    st.markdown(f"""
                    <div style='padding:1rem; border-radius:0.5rem; border-left:0.5rem solid {color}; 
                                background-color:#f8f9fa; margin-bottom:1rem;'>
                        <h4 style='margin:0; color:{color};'>{group}</h4>
                        <p style='margin:0.5rem 0;'><b>{details['numeric_value']}%</b> lower vaccination rate</p>
                        <p style='margin:0;'>{details['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    @staticmethod
    def show_factors(recommendations: Dict):
        st.header("🧠 Key Behavioral & Medical Drivers")
        
        # Behavioral Factors
        if behavioral := recommendations.get("Behavioral Factors"):
            st.subheader("Psychological Drivers")
            for factor, details in behavioral.items():
                # Safely get direction with default value
                direction = details.get('direction', 'Positive' if details.get('numeric_value', 0) >= 0 else 'Negative')
                
                with st.expander(f"{factor} ({direction} Impact)"):
                    col1, col2 = st.columns([3,1])
                    col1.metric("Correlation Strength", f"{details.get('numeric_value', 0):.2f}")
                    col2.metric("Priority", details.get('priority', 'Medium'))
                    
                    # Create gauge chart with proper error handling
                    try:
                        fig = go.Figure()
                        fig.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=abs(float(details.get('numeric_value', 0))),
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "lightgray"},
                                    {'range': [0.3, 0.6], 'color': "gray"},
                                    {'range': [0.6, 1], 'color': "darkgray"}
                                ]
                            }
                        ))
                        fig.update_layout(
                            title=f"Impact Strength: {factor}",
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create visualization: {str(e)}")
                    
                    st.info(f"💡 **Recommended Action**: {details.get('action', 'No action specified')}")

        # Medical Factors
        if medical := recommendations.get("Medical Factors"):
            st.subheader("Healthcare Leverage Points")
            for factor, details in medical.items():
                with st.container():
                    st.markdown(f"#### {factor}")
                    cols = st.columns(3)
                    cols[0].metric("Impact Size", f"+{details.get('numeric_value', 0)}%")
                    cols[1].metric("High Group", details.get('high_group', 'N/A'))
                    cols[2].metric("Low Group", details.get('low_group', 'N/A'))
                    
                    # Safe progress bar with value clamping
                    progress_value = min(1.0, max(0.0, float(details.get('numeric_value', 0))/100))
                    st.progress(progress_value)
                    st.info(f"🎯 **Action**: {details.get('action', 'No action specified')}")
                    st.markdown("---")

        if not behavioral and not medical:
            st.info("No significant behavioral or medical factors identified above threshold.")
    @staticmethod
    def show_barrier_messages(recommendations: Dict, df: pd.DataFrame):
        st.header("📲 Precision Messaging Campaign")
        st.markdown("Personalized SMS messages tailored to specific barriers")
        
        if not (barriers := recommendations.get("Barrier Messages")):
            st.warning("No barrier profiles available for messaging")
            return
            
        # Vaccine type selection
        vaccine_type = st.radio("Vaccine Campaign:", ["H1N1", "Seasonal"], 
                               horizontal=True, index=1, key="campaign_type")
        
        # Campaign statistics
        total_contacts = sum(len(d['affected_contacts']) for d in barriers.values())
        st.success(f"✨ **Campaign Reach**: {total_contacts:,} contacts across {len(barriers)} barrier types")
        
        # Process each barrier
        for idx, (title, details) in enumerate(barriers.items()):
            with st.expander(f"🚩 {title} ({details['priority']} Priority)"):
                # Barrier stats
                cols = st.columns(3)
                cols[0].metric("People Affected", details['numeric_value'])
                cols[1].metric("Priority", details['priority'])
                cols[2].metric("Ready Messages", len(details['affected_contacts']))
                
                # Message editing
                current_msg = details['action'].split("Seasonal: ")[1] if vaccine_type == "Seasonal" else details['action'].split("H1N1: ")[1].split("\n\n")[0]
                edited_msg = st.text_area(f"Edit {vaccine_type} message template:", 
                                        value=current_msg,
                                        height=150,
                                        key=f"msg_{idx}")
                
                # Sample contact preview
                if contact := details.get('sample_contact'):
                    st.markdown("**👤 Sample Contact**")
                    st.json(contact)
                    st.markdown("**📝 Message Preview**")
                    st.info(edited_msg.format(name=contact['name']))
                
                # Message sending
                if details['affected_contacts']:
                    if st.button(f"💬 Send to {len(details['affected_contacts'])} contacts", 
                               key=f"send_{idx}",
                               type="primary"):
                        Dashboard._send_messages(
                            details['affected_contacts'],
                            edited_msg,
                            vaccine_type,
                            details['barrier_type']
                        )
                else:
                    st.warning("No contacts available for this barrier")

    @staticmethod
    def _send_messages(contacts: List[Dict], template: str, campaign_type: str, barrier: str):
        progress_bar = st.progress(0)
        status = st.empty()
        results = {'sent': 0, 'failed': 0}
        
        for i, contact in enumerate(contacts):
            try:
                message = client.messages.create(
                    body=template.format(name=contact['name']),
                    to=contact['phone_number'],
                    from_=from_number
                )
                results['sent'] += 1
            except Exception as e:
                results['failed'] += 1
                st.error(f"Failed to send to {contact['phone_number']}: {str(e)}")
                
            progress = (i + 1) / len(contacts)
            progress_bar.progress(progress)
            status.text(f"Progress: {i+1}/{len(contacts)} | Sent: {results['sent']} | Failed: {results['failed']}")
        
        st.balloons()
        st.success(f"✅ Campaign complete! {results['sent']} {campaign_type} messages sent for {barrier} barrier")
        st.session_state['last_campaign'] = {
            'type': campaign_type,
            'barrier': barrier,
            'count': results['sent']
        }

    @staticmethod
    def show_analysis_report(analysis: Dict, recommendations: Dict):
        st.header("📊 Comprehensive Analysis Report")
        
        # Dataset statistics
        if stats := analysis.get('dataset_stats'):
            cols = st.columns(4)
            cols[0].metric("Total Records", stats['total_records'])
            cols[1].metric("H1N1 Vaccination", f"{stats['vaccination_rate_h1n1']*100:.1f}%")
            cols[2].metric("Seasonal Vaccination", f"{stats['vaccination_rate_seasonal']*100:.1f}%")
            cols[3].metric("Barriers Identified", len(analysis.get('barrier_profiles', [])))
        
        # Feature importance visualization
        features = []
        impacts = []
        categories = []
        
        for cat in ["Behavioral Factors", "Medical Factors"]:
            for details in recommendations.get(cat, {}).values():
                features.append(details['action'].split('targeting ')[-1].split('about ')[-1])
                impacts.append(details['numeric_value'])
                categories.append(cat.split()[0])
        
        if features:
            fig = px.bar(
                x=impacts,
                y=features,
                color=categories,
                orientation='h',
                title="Key Drivers of Vaccination Behavior",
                labels={'x': 'Impact Score', 'y': ''},
                color_discrete_map={'Behavioral': '#1f77b4', 'Medical': '#ff7f0e'}
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def setup_export(analysis: Dict, recommendations: Dict):
        st.sidebar.header("📤 Export Data")
        
        # Create export dataframe
        export_data = []
        for category, items in recommendations.items():
            for name, details in items.items():
                export_data.append({
                    'Category': category,
                    'Name': name,
                    'Impact Score': details.get('numeric_value', 0),
                    'Priority': details.get('priority', 'Medium'),
                    'Recommended Action': details.get('action', ''),
                    'Insight': details.get('insight', '')
                })
        
        df = pd.DataFrame(export_data)
        
        # Export options
        st.sidebar.download_button(
            "Download Full Report (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="vaccine_recommendations.csv",
            mime="text/csv"
        )
        
        if 'last_campaign' in st.session_state:
            campaign = st.session_state['last_campaign']
            st.sidebar.success(f"Last campaign: {campaign['count']} {campaign['type']} messages for {campaign['barrier']}")

# ---------------- Main Application ----------------
def main():
    configure_page()
    
    # Check for data
    if "results_df" not in st.session_state:
        st.warning("Please upload and process data first")
        st.stop()
        
    df = st.session_state["results_df"]
    
    # Analysis pipeline
    with st.spinner("🔍 Analyzing vaccination patterns..."):
        analysis = VaccineAnalyzer.analyze_data(df)
        recommendations = RecommendationEngine.generate_recommendations(analysis)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧑 Target Groups", 
        "📈 Key Factors", 
        "✉️ Messaging", 
        "📊 Full Report"
    ])
    
    with tab1:
        Dashboard.show_priority_groups(recommendations)
    
    with tab2:
        Dashboard.show_factors(recommendations)
    
    with tab3:
        Dashboard.show_barrier_messages(recommendations, df)
    
    with tab4:
        Dashboard.show_analysis_report(analysis, recommendations)
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()