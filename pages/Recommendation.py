import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional
from faker import Faker
import africastalking

# Initialize Faker
fake = Faker()

# Initialize Africa's Talking
username = st.secrets["africastalking"]["username"]
api_key = st.secrets["africastalking"]["api_key"]
sender_id = st.secrets["africastalking"]["sender_id"]

africastalking.initialize(username, api_key)
sms = africastalking.SMS

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
            'barrier_profiles': []   # list of dicts with messages and profile info
        }

        # Ensure string columns are strings (safe)
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
        columns_to_drop = ["employment_industry", "employment_occupation", "hhs_geo_region", "census_msa"]
        filtered_cols = [col for col in categorical_cols if col not in columns_to_drop]

        for col in filtered_cols:
            # ignore very high cardinality columns
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
            # skip high-cardinality
            if df[col].nunique() >= 5:
                continue

            # compute simple effect size: difference between groups if possible
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
        # Focus on those predicted not to vaccinate
        if 'h1n1_vaccine_pred' in df.columns and 'seasonal_vaccine_pred' in df.columns:
            df_target = df[(df['h1n1_vaccine_pred'] == 0) | (df['seasonal_vaccine_pred'] == 0)].copy()
        else:
            df_target = df.copy()

        # Create barrier flags
        df_target['barrier_no_insurance'] = df_target.get('health_insurance', pd.Series(0, index=df_target.index)).fillna(0).astype(float) == 0
        df_target['barrier_low_risk'] = df_target.get('opinion_h1n1_risk', pd.Series(3, index=df_target.index)).fillna(3) <= 2
        df_target['barrier_low_vaccine_belief'] = df_target.get('opinion_h1n1_vacc_effective', pd.Series(3, index=df_target.index)).fillna(3) <= 2
        df_target['barrier_low_knowledge'] = df_target.get('h1n1_knowledge', pd.Series(2, index=df_target.index)).fillna(2) <= 1
        df_target['barrier_access'] = df_target.get('behavioral_antiviral_meds', pd.Series(0, index=df_target.index)).fillna(0) == 0
        df_target['barrier_low_behaviors'] = df_target.get('safe_behavior_score', pd.Series(10, index=df_target.index)).fillna(10) <= 2

        # Compose a readable profile label from flags
        def compose_profile(row):
            parts = []
            if row['barrier_no_insurance']: parts.append("No Insurance")
            if row['barrier_low_vaccine_belief']: parts.append("Low Vaccine Belief")
            if row['barrier_low_risk']: parts.append("Low Risk Perception")
            if row['barrier_low_knowledge']: parts.append("Low Knowledge")
            if row['barrier_access']: parts.append("Access/Time Issues")
            if row['barrier_low_behaviors']: parts.append("Low Safe Behaviors")
            return " + ".join(sorted(parts)) if parts else "No Major Barrier"

        df_target['barrier_profile'] = df_target.apply(compose_profile, axis=1)

        # Count profiles
        profile_counts = df_target['barrier_profile'].value_counts().reset_index()
        profile_counts.columns = ['barrier_profile', 'people_affected']

        # Message templates per barrier
        barrier_messages = {
            'No Insurance': {
                'h1n1': "Hi {name}, the H1N1 vaccine is free for everyone. No insurance required — protect yourself today.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is free for everyone. No insurance needed — get your shot."
            },
            'Low Risk Perception': {
                'h1n1': "Hi {name}, H1N1 can infect healthy people too and spreads easily. Vaccination helps protect you and your family.",
                'seasonal': "Hi {name}, seasonal flu often affects healthy adults. The shot reduces your chance of severe illness."
            },
            'Low Vaccine Belief': {
                'h1n1': "Hi {name}, the H1N1 vaccine has been tested for safety and reduces serious illness — it's recommended by health experts.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is safe and effective. It greatly lowers hospital visits."
            },
            'Low Knowledge': {
                'h1n1': "Hi {name}, learn the basics: H1N1 spreads through droplets — vaccination is the best preventive measure.",
                'seasonal': "Hi {name}, flu can be serious. Ask your local clinic about the seasonal vaccine and where to get it."
            },
            'Access/Time Issues': {
                'h1n1': "Hi {name}, the H1N1 shot takes under 10 minutes and is available at nearby clinics — you often don't need an appointment.",
                'seasonal': "Hi {name}, getting the seasonal shot is quick and convenient — many clinics accept walk-ins."
            },
            'Low Safe Behaviors': {
                'h1n1': "Hi {name}, you already take some precautions; adding the H1N1 vaccine gives stronger protection.",
                'seasonal': "Hi {name}, your good habits help — the seasonal shot adds an important layer of defence."
            },
            'No Major Barrier': {
                'h1n1': "Hi {name}, getting vaccinated helps protect you and your community. Find your nearest clinic today.",
                'seasonal': "Hi {name}, getting the seasonal vaccine is a simple step to stay healthy — visit your local clinic."
            }
        }

        # Build output list
        profiles_output = []
        seen_profiles = set()
        for _, row in profile_counts.iterrows():
            profile = row['barrier_profile']
            count = int(row['people_affected'])

            # Ensure unique profiles
            if profile in seen_profiles:
                continue
            seen_profiles.add(profile)

            primary = profile.split(' + ')[0] if profile != "No Major Barrier" else "No Major Barrier"
            if primary not in barrier_messages:
                primary = "No Major Barrier"

            samples = []
            sample_rows = df_target[df_target['barrier_profile'] == profile].head(5)
            for _, r in sample_rows.iterrows():
                samples.append({
                    'fake_name': fake.first_name(),
                    'fake_phone': fake.phone_number()
                })

            profiles_output.append({
                'barrier_profile': profile,
                'people_affected': count,
                'primary_barrier': primary,
                'h1n1_message': barrier_messages[primary]['h1n1'],
                'seasonal_message': barrier_messages[primary]['seasonal'],
                'samples': samples,
                'priority': 'High' if primary in ['No Insurance', 'Low Vaccine Belief'] else 'Medium'
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
        for i, prof in enumerate(barrier_profiles[:500]):  # limit for display/export
            key = f"Barrier Profile: {prof['barrier_profile']}"
            recommendations[key] = {
                "insight": f"{prof['people_affected']} people with primary barrier: {prof['primary_barrier']}",
                "numeric_value": prof['people_affected'],
                "action": f"H1N1: {prof['h1n1_message']}\n\nSeasonal: {prof['seasonal_message']}",
                "priority": prof['priority']
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

        # Behavioral Factors
        if recommendations.get("Behavioral Factors"):
            st.subheader("Behavioral Drivers")
            for factor, details in recommendations.get("Behavioral Factors", {}).items():
                st.markdown(f"**{factor.title()}**")
                # show simple progress - clamp to [0,1]
                score = min(1.0, float(details.get('numeric_value', 0)))
                st.progress(score)
                st.caption(details.get('insight', ''))
                st.info(f"Action: {details.get('action', '')}")
        else:
            st.info("No behavioral drivers above threshold found.")

        # Medical Factors
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

        # 1. Verify required data exists
        if 'name' not in df.columns or 'phone_number' not in df.columns:
            st.error("Error: Missing required columns 'name' and/or 'phone_number'")
            return

        barrier_recs = recommendations.get("Barrier Messages", {})
        if not barrier_recs:
            st.warning("No barrier messages available.")
            return

        # 2. Vaccine type selection
        vaccine_type = st.radio("Select Vaccine Type:", 
                            ["H1N1", "Seasonal", "Both"],
                            horizontal=True, index=2)

        for idx, (key, details) in enumerate(barrier_recs.items()):
            with st.expander(f"Barrier {idx+1}: {details.get('insight', '')}"):
                # 3. Get message templates - FIXED ACCESS METHOD
                h1n1_template = details.get('h1n1_message', 
                                        "Hi {name}, H1N1 vaccine information")
                seasonal_template = details.get('seasonal_message', 
                                            "Hi {name}, Seasonal vaccine information")

                # 4. Make messages editable
                h1n1_msg = st.text_area("H1N1 Message:", h1n1_template,
                                    key=f"h1n1_{idx}")
                seasonal_msg = st.text_area("Seasonal Message:", seasonal_template,
                                        key=f"seasonal_{idx}")

                # 5. Prepare messages with validation
                messages = []
                sample_contacts = df[['name', 'phone_number']].head(3)  # Preview first 3
                
                for _, contact in sample_contacts.iterrows():
                    try:
                        # Format phone number
                        phone = str(contact['phone_number']).strip()
                        if phone.startswith('0') and len(phone) == 10:
                            phone = f"254{phone[1:]}"
                        elif len(phone) == 9 and phone[0] == '7':
                            phone = f"254{phone}"
                        
                        # Validate phone
                        if not (len(phone) == 12 and phone.startswith('254') and phone[3] in ['1', '7']):
                            continue

                        # Prepare message content - FIXED FORMATTING
                        name = contact['name']
                        final_msg = ""
                        
                        if vaccine_type in ["H1N1", "Both"]:
                            h1n1_final = h1n1_msg.replace("{name}", name) if "{name}" in h1n1_msg else h1n1_msg
                            final_msg += h1n1_final
                            
                        if vaccine_type in ["Seasonal", "Both"]:
                            seasonal_final = seasonal_msg.replace("{name}", name) if "{name}" in seasonal_msg else seasonal_msg
                            if final_msg:  # If Both, add separator
                                final_msg += "\n\n" + seasonal_final
                            else:
                                final_msg = seasonal_final
                        
                        if not final_msg.strip():
                            continue
                            
                        messages.append({
                            'to': f"+{phone}",
                            'name': name,
                            'message': final_msg
                        })
                        
                    except Exception as e:
                        st.error(f"Error preparing message: {str(e)}")
                        continue

                # 6. Show message preview
                if messages:
                    st.subheader("Message Preview")
                    preview_df = pd.DataFrame([{
                        'Recipient': msg['name'],
                        'Phone': msg['to'],
                        'Message Preview': msg['message'][:50] + "..." if len(msg['message']) > 50 else msg['message']
                    } for msg in messages[:3]])  # Show max 3 previews
                    st.dataframe(preview_df)
                else:
                    st.warning("No valid messages could be prepared")

                # 7. Send messages with proper validation
                if st.button(f"📤 Send Messages for {key}", key=f"send_{idx}"):
                    if not messages:
                        st.error("No valid messages to send")
                    else:
                        sent = 0
                        failed = 0
                        
                        for msg in messages:
                            try:
                                # Validate message isn't empty
                                if not msg['message'].strip():
                                    st.error(f"Empty message for {msg['to']}")
                                    failed += 1
                                    continue
                                    
                                # Send message
                                response = sms.send(
                                    message=msg['message'],
                                    recipients=[msg['to']],
                                    sender_id=sender_id
                                )
                                
                                # Verify response
                                if (isinstance(response, dict) and 
                                    response.get('SMSMessageData', {}).get('Recipients')):
                                    status = response['SMSMessageData']['Recipients'][0].get('status')
                                    if status == "Success":
                                        sent += 1
                                    else:
                                        error = response['SMSMessageData']['Recipients'][0].get('message', 'Unknown error')
                                        st.error(f"Failed to {msg['to']}: {error}")
                                        failed += 1
                                else:
                                    st.error(f"Invalid response for {msg['to']}")
                                    failed += 1
                                    
                            except Exception as e:
                                st.error(f"Send failed for {msg['to']}: {str(e)}")
                                failed += 1
                        
                        # Show results
                        col1, col2 = st.columns(2)
                        col1.success(f"✅ Sent: {sent}")
                        col2.error(f"❌ Failed: {failed}")
                        st.balloons()  # Celebrate success    

    @staticmethod
    def show_analysis_report(analysis: Dict, recommendations: Dict):
        st.header("Complete Analysis Report")

        # Feature Importance Visualization (simple)
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

    # Expect upstream page to put processed df in session_state
    if "results_df" not in st.session_state:
        st.warning("Please process data on the Home page first and place the results in st.session_state['results_df'].")
        st.stop()

    df = st.session_state["results_df"]

    # Run analysis
    with st.spinner("Analyzing vaccination data and building recommendations..."):
        analysis = VaccineAnalyzer.analyze_data(df)
        recommendations = RecommendationEngine.generate_recommendations(analysis)

    # Tabs
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