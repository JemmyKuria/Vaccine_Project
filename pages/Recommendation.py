import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Optional
from faker import Faker

# Optional import for Twilio only when sending
# from twilio.rest import Client

# Init Faker
fake = Faker()

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
        """
        Detect barriers, create barrier profiles, produce per-profile messages.
        Returns a list of dicts: each dict includes profile string, count, and message templates.
        """

        # Work on a copy
        df = df.copy()

        # If predictions exist use them, otherwise fall back to labels if available
        pred_col_h1 = 'h1n1_vaccine_pred' if 'h1n1_vaccine_pred' in df.columns else 'h1n1_label' if 'h1n1_label' in df.columns else None
        pred_col_seas = 'seasonal_vaccine_pred' if 'seasonal_vaccine_pred' in df.columns else 'seasonal_label' if 'seasonal_label' in df.columns else None

        # Focus on those predicted not to vaccinate for either vaccine (if preds available)
        if pred_col_h1 and pred_col_seas:
            df_target = df[(df[pred_col_h1] == 0) | (df[pred_col_seas] == 0)].copy()
        elif pred_col_h1:
            df_target = df[df[pred_col_h1] == 0].copy()
        elif pred_col_seas:
            df_target = df[df[pred_col_seas] == 0].copy()
        else:
            # if no prediction columns, consider entire df
            df_target = df.copy()

        # Create barrier flags (safe defaults if columns missing)
        df_target['barrier_no_insurance'] = df_target.get('health_insurance', pd.Series(0, index=df_target.index)).fillna(0).astype(float) == 0
        df_target['barrier_low_risk'] = df_target.get('opinion_h1n1_risk', pd.Series(3, index=df_target.index)).fillna(3) <= 2
        df_target['barrier_low_vaccine_belief'] = df_target.get('opinion_h1n1_vacc_effective', pd.Series(3, index=df_target.index)).fillna(3) <= 2
        df_target['barrier_low_knowledge'] = df_target.get('h1n1_knowledge', pd.Series(2, index=df_target.index)).fillna(2) <= 1
        # behavioral_antiviral_meds often 0/1 — use as proxy for access/time
        df_target['barrier_access'] = df_target.get('behavioral_antiviral_meds', pd.Series(0, index=df_target.index)).fillna(0) == 0
        # low safe behavior score if available
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
            return " + ".join(parts) if parts else "No Major Barrier"

        df_target['barrier_profile'] = df_target.apply(compose_profile, axis=1)

        # Count profiles
        profile_counts = df_target['barrier_profile'].value_counts().reset_index()
        profile_counts.columns = ['barrier_profile', 'people_affected']

        # Message templates per barrier (H1N1 and Seasonal)
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

        # Build output list: for each profile include message templates and a small sample
        profiles_output = []
        for _, row in profile_counts.iterrows():
            profile = row['barrier_profile']
            count = int(row['people_affected'])

            # pick primary barrier (first part) to choose message (simple heuristic)
            primary = profile.split(' + ')[0] if profile != "No Major Barrier" else "No Major Barrier"
            if primary not in barrier_messages:
                primary = "No Major Barrier"

            # Create a few fake contacts for preview (3 samples)
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
                # priority heuristic
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
    def show_barrier_messages(recommendations: Dict):
        st.header("📨 Personalized Messaging Recommendations")

        barrier_recs = recommendations.get("Barrier Messages", {})
        if not barrier_recs:
            st.warning("No barrier messages available.")
            return

        # Vaccine type selector
        vaccine_type = st.radio("Select Vaccine Type:", ["H1N1", "Seasonal", "Both"], horizontal=True, index=2)

        # Number of templates to preview
        n_preview = st.slider("Number of message templates to preview:", 1, 10, 5)

        st.subheader(f"Top {n_preview} Message Templates (preview)")

        # Convert recommendations dict to list sorted by numeric_value (descending)
        sorted_barriers = sorted(barrier_recs.items(), key=lambda x: x[1].get('numeric_value', 0), reverse=True)
        preview = sorted_barriers[:n_preview]

        # Create a send area
        send_mode = st.radio("Message mode:", ["Simulate only (no SMS)", "Send SMS (requires Twilio creds)"], index=0)
        send_button = st.button("Run Messaging for Previewed Templates")

        # Prepare messages list for possible sending
        messages_to_send = []

        for idx, (key, details) in enumerate(preview):
            with st.expander(f"Template {idx+1}: {details.get('insight', '')}"):
                primary_barrier = details.get('insight', '').replace('Detected barrier: ', '')
                st.markdown(f"**Barrier**: {primary_barrier}")
                st.markdown(f"**People affected (approx)**: {details.get('numeric_value', 0)}")
                # extract messages from action text
                action_text = details.get('action', '')
                # action_text formatted: "H1N1: <msg>\n\nSeasonal: <msg>"
                h1_msg = ""
                s_msg = ""
                if "H1N1:" in action_text and "Seasonal:" in action_text:
                    try:
                        h1_msg = action_text.split("H1N1: ")[1].split("\n\nSeasonal: ")[0].strip()
                        s_msg = action_text.split("\n\nSeasonal: ")[1].strip()
                    except Exception:
                        h1_msg = action_text
                        s_msg = action_text
                else:
                    # fallback
                    h1_msg = action_text
                    s_msg = action_text

                # Show messages according to vaccine_type
                if vaccine_type in ["H1N1", "Both"]:
                    st.info(f"**H1N1 message sample:** {h1_msg}")
                if vaccine_type in ["Seasonal", "Both"]:
                    st.success(f"**Seasonal message sample:** {s_msg}")

                # create fake contacts for this template (3 examples)
                examples = []
                for _ in range(3):
                    name = fake.first_name()
                    phone = fake.phone_number()
                    examples.append({'name': name, 'phone': phone,
                                     'h1n1_msg': h1_msg.format(name=name) if isinstance(h1_msg, str) else "",
                                     'seasonal_msg': s_msg.format(name=name) if isinstance(s_msg, str) else ""})
                st.table(pd.DataFrame(examples))

                # If user clicked to run messaging, prepare send list
                if send_button:
                    # For simulation we will take up to 10 fake recipients per template
                    for _ in range(min(10, int(max(1, details.get('numeric_value', 0) // 1000 + 1)))):
                        name = fake.first_name()
                        phone = fake.phone_number()
                        msg_text = ""
                        if vaccine_type == "H1N1":
                            msg_text = details.get('action', '').split("H1N1: ")[1].split("\n\nSeasonal: ")[0].format(name=name)
                        elif vaccine_type == "Seasonal":
                            msg_text = details.get('action', '').split("\n\nSeasonal: ")[1].format(name=name)
                        else:
                            # Both -> send both combined
                            h = details.get('action', '').split("H1N1: ")[1].split("\n\nSeasonal: ")[0].format(name=name)
                            s = details.get('action', '').split("\n\nSeasonal: ")[1].format(name=name)
                            msg_text = h + "\n\n" + s
                        messages_to_send.append({'to': phone, 'name': name, 'text': msg_text})

        # If user asked to run messaging
        if send_button:
            if not messages_to_send:
                st.warning("No messages prepared to send.")
            else:
                if send_mode.startswith("Simulate"):
                    st.success(f"Simulated sending {len(messages_to_send)} messages.")
                    for m in messages_to_send[:10]:
                        st.text(f"To: {m['name']} ({m['to']}) -> {m['text']}")
                else:
                    # Send via Twilio - read credentials from st.secrets
                    sid = st.secrets.get('twilio', {}).get('account_sid')
                    token = st.secrets.get('twilio', {}).get('auth_token')
                    from_number = st.secrets.get('twilio', {}).get('from_number')
                    if not sid or not token or not from_number:
                        st.error("Twilio credentials not found in st.secrets. Please add twilio.account_sid, twilio.auth_token, twilio.from_number.")
                    else:
                        try:
                            from twilio.rest import Client
                            client = Client(sid, token)
                            sent = 0
                            failed = 0
                            for m in messages_to_send:
                                try:
                                    client.messages.create(body=m['text'], from_=from_number, to=m['to'])
                                    sent += 1
                                except Exception as e:
                                    failed += 1
                            st.success(f"Sent: {sent} messages; Failed: {failed}")
                        except Exception as e:
                            st.error(f"Error sending messages: {e}")

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

        st.sidebar.download_button("📥 Download Full Report (JSON)", data=df.to_json(orient='records'),
                                   file_name="vaccine_recommendations.json")
        st.sidebar.download_button("📊 Executive Summary (CSV)", data=df.to_csv(index=False),
                                   file_name="vaccine_recommendations.csv")

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
        Dashboard.show_barrier_messages(recommendations)

    with tab4:
        Dashboard.show_analysis_report(analysis, recommendations)

    with tab5:
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()
