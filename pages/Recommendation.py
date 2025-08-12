import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List
from twilio.rest import Client

# Access Twilio secrets
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
from_number = st.secrets["twilio"]["from_number"]

# Initialize Twilio client
client = Client(account_sid, auth_token)

# ---------------- Page Configuration ----------------
def configure_page():
    st.set_page_config(page_title="AI Recommendation Engine", layout="wide")
    st.title("🤖 AI-Powered Vaccine Recommendations")

# ---------------- Analysis & Barrier Logic ----------------
class VaccineAnalyzer:
    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze dataset and identify key patterns (including barrier profiles)"""
        analysis = {
            'high_risk_groups': {},
            'behavior_factors': {},
            'medical_factors': {},
            'barrier_profiles': []
        }

        # Ensure string columns are strings
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # 1. Identify high-risk groups
        analysis['high_risk_groups'] = VaccineAnalyzer._find_high_risk_groups(df, categorical_cols)

        # 2. Behavior factors
        behavior_cols = [c for c in df.columns if any(x in c.lower() for x in ['opinion', 'behavior'])]
        analysis['behavior_factors'] = VaccineAnalyzer._analyze_factors(df, behavior_cols, correlation_threshold=0.2)

        # 3. Medical factors
        medical_cols = [c for c in df.columns if any(x in c.lower() for x in ['doctor', 'health'])]
        analysis['medical_factors'] = VaccineAnalyzer._analyze_medical_factors(df, medical_cols)

        # 4. Barriers and messaging
        analysis['barrier_profiles'] = VaccineAnalyzer._analyze_barriers(df)

        return analysis

    @staticmethod
    def _find_high_risk_groups(df: pd.DataFrame, categorical_cols: List[str]) -> Dict:
        """Identify groups with low vaccination rates"""
        high_risk = {}
        columns_to_drop = ["employment_industry", "employment_occupation", "hhs_geo_region", 
                          "census_msa", "name", "phone_number", "respondent_id", "h1n1_vaccine", 
                          "seasonal_vaccine", "h1n1_vaccine_pred", "seasonal_vaccine_pred", 
                          "h1n1_label", "seasonal_label", ""]
        filtered_cols = [col for col in categorical_cols if col not in columns_to_drop]

        for col in filtered_cols:
            if df[col].nunique() >= 20:
                continue

            if 'h1n1_label' in df.columns and 'seasonal_label' in df.columns:
                group_stats = df.groupby(col)[['h1n1_label', 'seasonal_label']].mean()
                high_risk_groups = group_stats[group_stats.mean(axis=1) < 0.4]
                if not high_risk_groups.empty:
                    high_risk[col] = high_risk_groups.mean(axis=1).sort_values().to_dict()
        return high_risk

    @staticmethod
    def _analyze_factors(df: pd.DataFrame, cols: List[str], correlation_threshold: float = 0.2) -> Dict:
        """Calculate correlation between factors and vaccination labels"""
        factors = {}
        if not {'h1n1_label', 'seasonal_label'}.issubset(set(df.columns)):
            return factors

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
        if 'h1n1_label' not in df.columns:
            return factors

        for col in cols:
            if df[col].nunique() >= 5:
                continue

            try:
                effect_size = df.groupby(col)['h1n1_label'].mean().diff().abs().max()
            except Exception:
                effect_size = None

            if effect_size is not None and not pd.isna(effect_size) and effect_size > 0.15:
                factors[col] = float(effect_size)
        return factors

    @staticmethod
    def _analyze_barriers(df: pd.DataFrame) -> List[Dict]:
        df = df.copy()
        if 'h1n1_vaccine_pred' in df.columns and 'seasonal_vaccine_pred' in df.columns:
            df_target = df[(df['h1n1_vaccine_pred'] == 0) | (df['seasonal_vaccine_pred'] == 0)].copy()
        else:
            df_target = df.copy()

        # Barrier priority order (higher priority first)
        barrier_conditions = [
            ('No Insurance', df_target.get('health_insurance', 1) == 0),
            ('Low Vaccine Belief', df_target.get('opinion_h1n1_vacc_effective', 3) <= 2),
            ('Low Risk Perception', df_target.get('opinion_h1n1_risk', 3) <= 2),
            ('Low Knowledge', df_target.get('h1n1_knowledge', 2) <= 1),
            ('Access Issues', df_target.get('behavioral_antiviral_meds', 0) == 0),
            ('Low Safe Behaviors', df_target.get('safe_behavior_score', 10) <= 2)
        ]

        # Assign each person to exactly one barrier
        df_target['barrier_profile'] = 'No Major Barrier'
        for barrier, condition in barrier_conditions:
            df_target.loc[condition & (df_target['barrier_profile'] == 'No Major Barrier'), 'barrier_profile'] = barrier

        # Message templates
        barrier_messages = {
            'No Insurance': {
                'h1n1': "Hi {name}, the H1N1 vaccine is free for everyone. No insurance required — protect yourself today.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is free for everyone. No insurance needed — get your shot."
            },
            'Low Risk Perception': {
                'h1n1': "Hi {name}, H1N1 can infect healthy people too. Vaccination helps protect you and your family.",
                'seasonal': "Hi {name}, seasonal flu often affects healthy adults. The shot reduces severe illness risk."
            },
            'Low Vaccine Belief': {
                'h1n1': "Hi {name}, the H1N1 vaccine is safe and effective — recommended by health experts.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is safe and greatly lowers hospital visits."
            },
            'Low Knowledge': {
                'h1n1': "Hi {name}, H1N1 spreads through droplets. Vaccination is the best preventive measure.",
                'seasonal': "Hi {name}, flu can be serious. Ask your clinic about the seasonal vaccine."
            },
            'Access Issues': {
                'h1n1': "Hi {name}, the H1N1 shot takes under 10 minutes at nearby clinics — no appointment needed.",
                'seasonal': "Hi {name}, getting the seasonal shot is quick — many clinics accept walk-ins."
            },
            'Low Safe Behaviors': {
                'h1n1': "Hi {name}, adding the H1N1 vaccine gives stronger protection with your precautions.",
                'seasonal': "Hi {name}, the seasonal shot adds an important layer of defence to your habits."
            },
            'No Major Barrier': {
                'h1n1': "Hi {name}, getting vaccinated helps protect you and your community.",
                'seasonal': "Hi {name}, the seasonal vaccine is a simple step to stay healthy."
            }
        }

        # Build profiles output
        profiles_output = []
        profile_counts = df_target['barrier_profile'].value_counts().reset_index()
        profile_counts.columns = ['barrier_profile', 'people_affected']

        for _, row in profile_counts.iterrows():
            profile = row['barrier_profile']
            count = int(row['people_affected'])
            
            # Get sample contact
            sample_contact = None
            if 'name' in df_target.columns and 'phone_number' in df_target.columns:
                sample_df = df_target[df_target['barrier_profile'] == profile]
                if not sample_df.empty:
                    sample_row = sample_df.iloc[0]
                    sample_contact = {
                        'name': sample_row['name'],
                        'phone': sample_row['phone_number']
                    }

            # Get all affected contacts
            affected_contacts = []
            if 'name' in df_target.columns and 'phone_number' in df_target.columns:
                affected_df = df_target[df_target['barrier_profile'] == profile]
                affected_contacts = affected_df[['name', 'phone_number']].to_dict('records')

            profiles_output.append({
                'barrier_profile': profile,
                'people_affected': count,
                'primary_barrier': profile,
                'h1n1_message': barrier_messages[profile]['h1n1'],
                'seasonal_message': barrier_messages[profile]['seasonal'],
                'sample_contact': sample_contact,
                'affected_contacts': affected_contacts,
                'priority': 'High' if profile in ['No Insurance', 'Low Vaccine Belief'] else 'Medium'
            })

        return profiles_output

# ---------------- Recommendation Engine ----------------
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
                "insight": f"{stats['direction']} correlation (r={abs(stats['correlation']):.2f})",
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

    @staticmethod
    def _generate_barrier_recommendations(analysis: Dict) -> Dict:
        recommendations = {}
        barrier_profiles = analysis.get('barrier_profiles', [])
        for prof in barrier_profiles[:500]:  # limit for display
            key = f"Barrier Profile: {prof['barrier_profile']}"
            recommendations[key] = {
                "insight": f"{prof['people_affected']} people with barrier: {prof['primary_barrier']}",
                "numeric_value": prof['people_affected'],
                "action": f"H1N1: {prof['h1n1_message']}\n\nSeasonal: {prof['seasonal_message']}",
                "priority": prof['priority'],
                "sample_contact": prof.get('sample_contact'),
                "affected_contacts": prof.get('affected_contacts', [])
            }
        return recommendations

# ---------------- Dashboard Components ----------------
class Dashboard:
    @staticmethod
    def show_priority_groups(recommendations: Dict):
        st.header("Priority Intervention Groups (Sorted by Risk)")
        target_groups = recommendations.get("Target Groups", {})
        sorted_groups = sorted(
            target_groups.items(),
            key=lambda x: x[1].get('numeric_value', 0),
            reverse=True
        )

        cols = st.columns(2)
        for i, (group, details) in enumerate(sorted_groups[:10]):
            with cols[i % 2]:
                priority = details.get('priority', 'Medium')
                color = "#ff4b4b" if priority == "High" else "#ffa44b"
                st.markdown(f"<h4 style='color:{color}'>🔴 {group} ({priority} Priority)</h4>", unsafe_allow_html=True)
                st.metric("Hesitancy", f"{details.get('numeric_value', 0)}%")
                st.markdown(f"**Why**: {details.get('insight', '')}")
                st.markdown(f"**Action**: {details.get('action', '')}")

    @staticmethod
    def show_factors(recommendations: Dict):
        st.header("Most Influential Factors")

        if recommendations.get("Behavioral Factors"):
            st.subheader("Behavioral Drivers")
            for factor, details in recommendations.get("Behavioral Factors", {}).items():
                st.markdown(f"**{factor.title()}**")
                score = min(1.0, float(details.get('numeric_value', 0)))
                st.progress(score)
                st.caption(details.get('insight', ''))
                st.info(f"Action: {details.get('action', '')}")
        else:
            st.info("No behavioral drivers above threshold found.")

        if recommendations.get("Medical Factors"):
            st.subheader("Healthcare Leverage Points")
            for factor, details in recommendations.get("Medical Factors", {}).items():
                cols = st.columns([3,1])
                cols[0].markdown(f"**{factor.title()}**")
                cols[0].caption(details.get('insight', ''))
                cols[0].info(f"Action: {details.get('action', '')}")
                cols[1].metric("Impact", f"+{details.get('numeric_value', 0)}%")
        else:
            st.info("No medical leverage points found.")

    @staticmethod
    def show_barrier_messages(recommendations: Dict, df: pd.DataFrame):
        st.header("📨 Personalized Messaging Recommendations")
        barrier_recs = recommendations.get("Barrier Messages", {})
        if not barrier_recs:
            st.warning("No barrier messages available.")
            return

        vaccine_type = st.radio("Select Vaccine Type:", ["H1N1", "Seasonal"], horizontal=True, index=0)
        st.subheader("Message Templates (editable)")

        for idx, (key, details) in enumerate(barrier_recs.items()):
            with st.expander(f"Barrier {idx + 1}: {details.get('insight', '')}"):
                barrier_name = details.get('primary_barrier', 'Unknown')
                st.markdown(f"**People affected**: {details.get('numeric_value', 0)}")
                
                # Get current messages
                h1_msg = details['action'].split("H1N1: ")[1].split("\n\nSeasonal: ")[0]
                s_msg = details['action'].split("\n\nSeasonal: ")[1]
                
                # Editable fields
                if vaccine_type == "H1N1":
                    h1_msg = st.text_area(f"H1N1 message for '{barrier_name}':", h1_msg, key=f"h1n1_{idx}")
                else:
                    s_msg = st.text_area(f"Seasonal message for '{barrier_name}':", s_msg, key=f"seasonal_{idx}")

                # Show sample contact
                sample_contact = details.get('sample_contact')
                if sample_contact:
                    st.markdown("**Sample Contact:**")
                    st.write(f"Name: {sample_contact['name']}")
                    st.write(f"Phone: {sample_contact['phone']}")
                    
                    # Preview message
                    msg = h1_msg if vaccine_type == "H1N1" else s_msg
                    preview = msg.format(name=sample_contact['name'])
                    st.markdown("**Message Preview:**")
                    st.info(preview)
                else:
                    st.warning("No sample contact available")

                # Prepare ALL messages for this barrier
                affected_contacts = details.get('affected_contacts', [])
                messages_to_send = []
                for contact in affected_contacts:
                    msg_text = (h1_msg if vaccine_type == "H1N1" else s_msg).format(name=contact['name'])
                    messages_to_send.append({
                        'to': contact['phone_number'],
                        'name': contact['name'],
                        'text': msg_text
                    })

                # Store messages in session state
                st.session_state[f"messages_barrier_{idx}"] = messages_to_send
                st.markdown(f"**Total messages prepared:** {len(messages_to_send)}")

                # Send button for ALL messages in this barrier
                if st.button(f"📤 Send ALL Messages for {barrier_name}", key=f"send_{idx}"):
                    msgs = st.session_state.get(f"messages_barrier_{idx}", [])
                    if not msgs:
                        st.warning("No messages prepared to send.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        sent, failed = 0, 0
                        
                        for i, m in enumerate(msgs):
                            try:
                                message = client.messages.create(
                                    body=m['text'],
                                    to=m['to'],
                                    from_=from_number
                                )
                                if message.sid:
                                    sent += 1
                                else:
                                    failed += 1
                                progress = (i + 1) / len(msgs)
                                progress_bar.progress(progress)
                                status_text.text(f"Sending {i+1}/{len(msgs)} - {sent} sent, {failed} failed")
                            except Exception as e:
                                failed += 1
                                st.error(f"Error sending to {m['to']}: {str(e)}")
                        
                        st.success(f"✅ Sent: {sent} messages; ❌ Failed: {failed}")
                        if sent > 0:
                            st.balloons()

    @staticmethod
    def show_analysis_report(analysis: Dict, recommendations: Dict):
        st.header("Complete Analysis Report")

        features = []
        importance = []

        for factor_type in ["Behavioral Factors", "Medical Factors"]:
            for details in recommendations.get(factor_type, {}).values():
                insight = details.get('insight', '')
                if 'r=' in insight:
                    try:
                        importance.append(abs(float(insight.split('r=')[1][:4])))
                    except Exception:
                        importance.append(details.get('numeric_value', 0))
                else:
                    importance.append(details.get('numeric_value', 0) / 100 if details.get('numeric_value', 0) else 0)
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

        with st.expander("📊 Full analysis JSON"):
            st.json(analysis)

    @staticmethod
    def setup_export(analysis: Dict, recommendations: Dict):
        st.sidebar.header("Export Options")

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

        st.sidebar.download_button(
            "📥 Download Full Report (JSON)",
            data=df.to_json(orient='records'),
            file_name="vaccine_recommendations.json"
        )
        st.sidebar.download_button(
            "📊 Executive Summary (CSV)",
            data=df.to_csv(index=False),
            file_name="vaccine_recommendations.csv"
        )

# ---------------- Main ----------------
def main():
    configure_page()

    if "results_df" not in st.session_state:
        st.warning("Please process data on the Home page first and place the results in st.session_state['results_df'].")
        st.stop()

    df = st.session_state["results_df"]

    with st.spinner("Analyzing vaccination data and building recommendations..."):
        analysis = VaccineAnalyzer.analyze_data(df)
        recommendations = RecommendationEngine.generate_recommendations(analysis)

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
        Dashboard.show_barrier_messages(recommendations, df)

    with tab4:
        Dashboard.show_analysis_report(analysis, recommendations)

    with tab5:
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()