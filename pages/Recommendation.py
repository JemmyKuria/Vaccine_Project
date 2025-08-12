import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
from twilio.rest import Client
import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Access Twilio secrets
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
from_number = st.secrets["twilio"]["from_number"]

# Initialize Twilio client
client = Client(account_sid, auth_token)

# ---------------- Page Configuration ----------------
def configure_page():
    st.set_page_config(
        page_title="AI Recommendation Engine", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "AI-Powered Vaccine Recommendation System v2.0"
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .priority-high { border-left-color: #ff4b4b !important; }
    .priority-medium { border-left-color: #ffa44b !important; }
    .priority-low { border-left-color: #00cc88 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>🤖 AI-Powered Vaccine Recommendations</h1><p>Advanced Analytics & Personalized Messaging System</p></div>', unsafe_allow_html=True)

# ---------------- Enhanced Analysis & Barrier Logic ----------------
class VaccineAnalyzer:
    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Dict]:
        """Enhanced analyze dataset with advanced patterns and ML insights"""
        analysis = {
            'high_risk_groups': {},
            'behavior_factors': {},
            'medical_factors': {},
            'barrier_profiles': [],
            'demographic_insights': {},
            'geographic_patterns': {},
            'ml_insights': {},
            'temporal_patterns': {},
            'data_quality': {}
        }

        # Data quality assessment
        analysis['data_quality'] = VaccineAnalyzer._assess_data_quality(df)

        # Ensure string columns are strings
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # Original analysis
        analysis['high_risk_groups'] = VaccineAnalyzer._find_high_risk_groups(df, categorical_cols)

        # Enhanced behavior factors with advanced correlation analysis
        behavior_cols = [c for c in df.columns if any(x in c.lower() for x in ['opinion', 'behavior'])]
        analysis['behavior_factors'] = VaccineAnalyzer._analyze_factors(df, behavior_cols, correlation_threshold=0.2)

        # Medical factors
        medical_cols = [c for c in df.columns if any(x in c.lower() for x in ['doctor', 'health'])]
        analysis['medical_factors'] = VaccineAnalyzer._analyze_medical_factors(df, medical_cols)

        # Enhanced barriers
        analysis['barrier_profiles'] = VaccineAnalyzer._analyze_barriers(df)

        # New enhanced analyses
        analysis['demographic_insights'] = VaccineAnalyzer._analyze_demographics(df)
        analysis['geographic_patterns'] = VaccineAnalyzer._analyze_geographic_patterns(df)
        analysis['ml_insights'] = VaccineAnalyzer._generate_ml_insights(df)
        analysis['temporal_patterns'] = VaccineAnalyzer._analyze_temporal_patterns(df)

        return analysis

    @staticmethod
    def _assess_data_quality(df: pd.DataFrame) -> Dict:
        """Assess data quality metrics"""
        return {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'completeness_rate': (1 - df.isnull().sum() / len(df)).mean(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }

    @staticmethod
    def _analyze_demographics(df: pd.DataFrame) -> Dict:
        """Analyze demographic patterns in vaccination behavior"""
        demographics = {}
        
        # Age group analysis
        if 'age_group' in df.columns and any(col in df.columns for col in ['h1n1_label', 'seasonal_label']):
            age_vaccination = df.groupby('age_group')[['h1n1_label', 'seasonal_label']].mean() if 'h1n1_label' in df.columns else {}
            demographics['age_patterns'] = age_vaccination.to_dict() if hasattr(age_vaccination, 'to_dict') else {}

        # Income analysis
        income_cols = [col for col in df.columns if 'income' in col.lower()]
        if income_cols and 'h1n1_label' in df.columns:
            for col in income_cols:
                demographics[f'{col}_impact'] = df.groupby(col)['h1n1_label'].mean().to_dict()

        # Education analysis
        education_cols = [col for col in df.columns if 'education' in col.lower()]
        if education_cols and 'h1n1_label' in df.columns:
            for col in education_cols:
                demographics[f'{col}_impact'] = df.groupby(col)['h1n1_label'].mean().to_dict()

        return demographics

    @staticmethod
    def _analyze_geographic_patterns(df: pd.DataFrame) -> Dict:
        """Analyze geographic vaccination patterns"""
        patterns = {}
        
        geo_cols = [col for col in df.columns if any(term in col.lower() for term in ['geo', 'region', 'state', 'msa'])]
        
        for col in geo_cols:
            if df[col].nunique() < 50 and 'h1n1_label' in df.columns:  # Avoid too granular data
                regional_rates = df.groupby(col)['h1n1_label'].agg(['mean', 'count']).reset_index()
                patterns[col] = regional_rates.to_dict('records')

        return patterns

    @staticmethod
    def _generate_ml_insights(df: pd.DataFrame) -> Dict:
        """Generate machine learning insights"""
        insights = {}
        
        if not {'h1n1_label', 'seasonal_label'}.issubset(df.columns):
            return {'error': 'Required target variables not found'}

        try:
            # Prepare features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ['h1n1_label', 'seasonal_label']]
            
            if len(feature_cols) < 5:
                return {'error': 'Insufficient numeric features for ML analysis'}

            X = df[feature_cols].fillna(df[feature_cols].median())
            y_h1n1 = df['h1n1_label']

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y_h1n1, test_size=0.2, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            insights['feature_importance'] = feature_importance.head(10).to_dict('records')
            insights['model_score'] = rf.score(X_test, y_test)
            
            # Predictions
            predictions = rf.predict(X_test)
            insights['classification_report'] = classification_report(y_test, predictions, output_dict=True)

        except Exception as e:
            insights['error'] = str(e)

        return insights

    @staticmethod
    def _analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns if date columns exist"""
        patterns = {}
        
        # Look for date-like columns
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'month', 'year'])]
        
        if date_cols:
            patterns['date_columns_found'] = date_cols
            # Add temporal analysis here if needed
        else:
            patterns['message'] = 'No temporal columns identified'

        return patterns

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
                    'direction': 'Negative' if corr < 0 else 'Positive',
                    'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
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
        """Enhanced barrier analysis with more sophisticated profiling"""
        df = df.copy()
        if 'h1n1_vaccine_pred' in df.columns and 'seasonal_vaccine_pred' in df.columns:
            df_target = df[(df['h1n1_vaccine_pred'] == 0) | (df['seasonal_vaccine_pred'] == 0)].copy()
        else:
            df_target = df.copy()

        # Enhanced barrier priority order
        barrier_conditions = [
            ('No Insurance', df_target.get('health_insurance', 1) == 0),
            ('Low Vaccine Belief', df_target.get('opinion_h1n1_vacc_effective', 3) <= 2),
            ('Low Risk Perception', df_target.get('opinion_h1n1_risk', 3) <= 2),
            ('Low Knowledge', df_target.get('h1n1_knowledge', 2) <= 1),
            ('Access Issues', df_target.get('behavioral_antiviral_meds', 0) == 0),
            ('Low Safe Behaviors', df_target.get('safe_behavior_score', 10) <= 2),
            ('Healthcare Distrust', df_target.get('doctor_recc_h1n1', 1) == 0),
            ('Concern About Safety', df_target.get('opinion_h1n1_sick_from_vacc', 2) >= 3)
        ]

        # Assign each person to exactly one barrier
        df_target['barrier_profile'] = 'No Major Barrier'
        for barrier, condition in barrier_conditions:
            df_target.loc[condition & (df_target['barrier_profile'] == 'No Major Barrier'), 'barrier_profile'] = barrier

        # Enhanced message templates with A/B testing variants
        barrier_messages = {
            'No Insurance': {
                'h1n1': "Hi {name}, the H1N1 vaccine is free for everyone. No insurance required — protect yourself today.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is free for everyone. No insurance needed — get your shot.",
                'variant_h1n1': "Hi {name}, worried about costs? H1N1 vaccination is completely free - no insurance or payment needed.",
                'variant_seasonal': "Hi {name}, flu shots are free regardless of insurance status. Protect yourself today."
            },
            'Low Risk Perception': {
                'h1n1': "Hi {name}, H1N1 can infect healthy people too. Vaccination helps protect you and your family.",
                'seasonal': "Hi {name}, seasonal flu often affects healthy adults. The shot reduces severe illness risk.",
                'variant_h1n1': "Hi {name}, even healthy adults can get seriously ill from H1N1. Vaccination is your best defense.",
                'variant_seasonal': "Hi {name}, don't let flu catch you off guard. Even mild cases can keep you down for weeks."
            },
            'Low Vaccine Belief': {
                'h1n1': "Hi {name}, the H1N1 vaccine is safe and effective — recommended by health experts.",
                'seasonal': "Hi {name}, the seasonal flu vaccine is safe and greatly lowers hospital visits.",
                'variant_h1n1': "Hi {name}, millions have safely received the H1N1 vaccine. It's proven to prevent serious illness.",
                'variant_seasonal': "Hi {name}, flu vaccines have protected families for decades. Trust the science."
            },
            'Healthcare Distrust': {
                'h1n1': "Hi {name}, talk to a healthcare provider you trust about H1N1 vaccination benefits.",
                'seasonal': "Hi {name}, get answers to your flu vaccine questions from a trusted medical professional.",
                'variant_h1n1': "Hi {name}, your health matters. Find a healthcare provider who listens to your H1N1 concerns.",
                'variant_seasonal': "Hi {name}, seek out healthcare providers who respect your questions about flu vaccination."
            },
            'Concern About Safety': {
                'h1n1': "Hi {name}, H1N1 vaccine side effects are typically mild and brief. Serious reactions are extremely rare.",
                'seasonal': "Hi {name}, flu vaccines are rigorously tested. Mild soreness is normal and shows your immunity is building.",
                'variant_h1n1': "Hi {name}, worried about H1N1 vaccine safety? Talk to your doctor about the minimal risks vs. benefits.",
                'variant_seasonal': "Hi {name}, millions safely get flu shots yearly. Severe side effects are far rarer than flu complications."
            }
        }

        # Add default messages for remaining barriers
        for barrier in ['Low Knowledge', 'Access Issues', 'Low Safe Behaviors', 'No Major Barrier']:
            if barrier not in barrier_messages:
                barrier_messages[barrier] = {
                    'h1n1': f"Hi {{name}}, the H1N1 vaccine is an important step in protecting your health.",
                    'seasonal': f"Hi {{name}}, seasonal flu vaccination helps keep you and your community healthy.",
                    'variant_h1n1': f"Hi {{name}}, consider getting your H1N1 vaccination to stay protected.",
                    'variant_seasonal': f"Hi {{name}}, don't miss your seasonal flu shot this year."
                }

        # Build enhanced profiles output
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

            # Enhanced priority scoring
            priority_score = VaccineAnalyzer._calculate_priority_score(profile, count, len(df_target))
            
            profiles_output.append({
                'barrier_profile': profile,
                'people_affected': count,
                'percentage': round(count / len(df_target) * 100, 1),
                'primary_barrier': profile,
                'h1n1_message': barrier_messages[profile]['h1n1'],
                'seasonal_message': barrier_messages[profile]['seasonal'],
                'h1n1_variant': barrier_messages[profile].get('variant_h1n1', barrier_messages[profile]['h1n1']),
                'seasonal_variant': barrier_messages[profile].get('variant_seasonal', barrier_messages[profile]['seasonal']),
                'sample_contact': sample_contact,
                'affected_contacts': affected_contacts,
                'priority': priority_score['level'],
                'priority_score': priority_score['score'],
                'estimated_response_rate': VaccineAnalyzer._estimate_response_rate(profile)
            })

        return sorted(profiles_output, key=lambda x: x['priority_score'], reverse=True)

    @staticmethod
    def _calculate_priority_score(barrier: str, count: int, total: int) -> Dict:
        """Calculate priority score for barriers"""
        base_weights = {
            'No Insurance': 0.9,
            'Low Vaccine Belief': 0.85,
            'Healthcare Distrust': 0.8,
            'Concern About Safety': 0.75,
            'Low Risk Perception': 0.7,
            'Low Knowledge': 0.6,
            'Access Issues': 0.55,
            'Low Safe Behaviors': 0.5,
            'No Major Barrier': 0.3
        }
        
        base_score = base_weights.get(barrier, 0.5)
        size_factor = min(count / total * 2, 0.3)  # Cap size influence
        final_score = base_score + size_factor
        
        if final_score >= 0.8:
            level = 'Critical'
        elif final_score >= 0.6:
            level = 'High'
        elif final_score >= 0.4:
            level = 'Medium'
        else:
            level = 'Low'
            
        return {'score': final_score, 'level': level}

    @staticmethod
    def _estimate_response_rate(barrier: str) -> float:
        """Estimate expected response rate based on barrier type"""
        rates = {
            'No Insurance': 0.75,
            'Access Issues': 0.70,
            'Low Knowledge': 0.65,
            'Low Risk Perception': 0.45,
            'Low Vaccine Belief': 0.25,
            'Healthcare Distrust': 0.20,
            'Concern About Safety': 0.30,
            'Low Safe Behaviors': 0.50,
            'No Major Barrier': 0.60
        }
        return rates.get(barrier, 0.45)

# ---------------- Enhanced Recommendation Engine ----------------
class RecommendationEngine:
    @staticmethod
    def generate_recommendations(analysis: Dict) -> Dict[str, Dict]:
        """Convert analysis into enhanced actionable recommendations"""
        return {
            "Target Groups": RecommendationEngine._generate_group_recommendations(analysis),
            "Behavioral Factors": RecommendationEngine._generate_behavior_recommendations(analysis),
            "Medical Factors": RecommendationEngine._generate_medical_recommendations(analysis),
            "Barrier Messages": RecommendationEngine._generate_barrier_recommendations(analysis),
            "ML Insights": RecommendationEngine._generate_ml_recommendations(analysis),
            "Geographic Patterns": RecommendationEngine._generate_geographic_recommendations(analysis),
            "Campaign Strategies": RecommendationEngine._generate_campaign_strategies(analysis)
        }

    @staticmethod
    def _generate_ml_recommendations(analysis: Dict) -> Dict:
        """Generate recommendations from ML insights"""
        recommendations = {}
        ml_insights = analysis.get('ml_insights', {})
        
        if 'feature_importance' in ml_insights:
            for feature in ml_insights['feature_importance'][:5]:
                feature_name = feature['feature'].replace('_', ' ').title()
                recommendations[f"ML Feature: {feature_name}"] = {
                    "insight": f"ML model identifies this as key predictor (importance: {feature['importance']:.3f})",
                    "numeric_value": feature['importance'],
                    "action": f"Focus interventions on {feature_name.lower()} factors",
                    "priority": "High" if feature['importance'] > 0.1 else "Medium"
                }
        
        if 'model_score' in ml_insights:
            recommendations["Model Performance"] = {
                "insight": f"Prediction accuracy: {ml_insights['model_score']:.1%}",
                "numeric_value": ml_insights['model_score'],
                "action": "Use model predictions to prioritize outreach",
                "priority": "High" if ml_insights['model_score'] > 0.8 else "Medium"
            }
            
        return recommendations

    @staticmethod
    def _generate_geographic_recommendations(analysis: Dict) -> Dict:
        """Generate geographic targeting recommendations"""
        recommendations = {}
        geo_patterns = analysis.get('geographic_patterns', {})
        
        for pattern_name, pattern_data in geo_patterns.items():
            if isinstance(pattern_data, list) and pattern_data:
                # Find regions with lowest vaccination rates
                sorted_regions = sorted(pattern_data, key=lambda x: x.get('mean', 1))
                for region in sorted_regions[:3]:  # Top 3 lowest
                    region_name = str(region.get(pattern_name, 'Unknown'))
                    rate = region.get('mean', 0)
                    count = region.get('count', 0)
                    
                    recommendations[f"Geographic Focus: {region_name}"] = {
                        "insight": f"Low vaccination rate: {rate:.1%} ({count} people)",
                        "numeric_value": 1 - rate,  # Convert to risk score
                        "action": f"Targeted campaign in {region_name}",
                        "priority": "High" if rate < 0.4 else "Medium"
                    }
        
        return recommendations

    @staticmethod
    def _generate_campaign_strategies(analysis: Dict) -> Dict:
        """Generate high-level campaign strategies"""
        strategies = {}
        barrier_profiles = analysis.get('barrier_profiles', [])
        
        if barrier_profiles:
            # Overall strategy based on top barriers
            top_barriers = sorted(barrier_profiles, key=lambda x: x['people_affected'], reverse=True)[:3]
            
            for i, barrier in enumerate(top_barriers):
                strategy_name = f"Strategy {i+1}: Address {barrier['barrier_profile']}"
                strategies[strategy_name] = {
                    "insight": f"Affects {barrier['people_affected']} people ({barrier.get('percentage', 0)}%)",
                    "numeric_value": barrier['people_affected'],
                    "action": RecommendationEngine._get_strategy_action(barrier['barrier_profile']),
                    "priority": barrier['priority'],
                    "estimated_response": f"{barrier.get('estimated_response_rate', 0.5):.0%}"
                }
        
        return strategies

    @staticmethod
    def _get_strategy_action(barrier: str) -> str:
        """Get strategic action for each barrier type"""
        strategies = {
            'No Insurance': "Partner with free clinics, emphasize no-cost messaging",
            'Low Vaccine Belief': "Deploy trusted community leaders, share success stories",
            'Healthcare Distrust': "Build relationships with community health workers",
            'Concern About Safety': "Provide detailed safety data, address specific concerns",
            'Low Risk Perception': "Share real stories, emphasize community protection",
            'Low Knowledge': "Educational campaigns, FAQ resources",
            'Access Issues': "Mobile clinics, extended hours, workplace programs",
            'Low Safe Behaviors': "Integrate with broader health promotion",
            'No Major Barrier': "Simple reminders and convenience messaging"
        }
        return strategies.get(barrier, "Develop targeted intervention strategy")

    # Keep existing methods
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
                "insight": f"{stats['direction']} correlation (r={abs(stats['correlation']):.2f}, {stats.get('strength', 'Moderate')})",
                "numeric_value": abs(stats['correlation']),
                "action": f"Campaign focusing on {factor.replace('_', ' ')}",
                "priority": "High" if stats.get('strength') == 'Strong' else "Medium"
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
                "insight": f"{prof['people_affected']} people ({prof.get('percentage', 0)}%) with barrier: {prof['primary_barrier']}",
                "numeric_value": prof['people_affected'],
                "action": f"H1N1: {prof['h1n1_message']}\n\nSeasonal: {prof['seasonal_message']}",
                "priority": prof['priority'],
                "sample_contact": prof.get('sample_contact'),
                "affected_contacts": prof.get('affected_contacts', []),
                "estimated_response_rate": prof.get('estimated_response_rate', 0.5),
                "priority_score": prof.get('priority_score', 0.5)
            }
        return recommendations

# ---------------- Enhanced Dashboard Components ----------------
class Dashboard:
    @staticmethod
    def show_overview_metrics(analysis: Dict, recommendations: Dict):
        """Display key overview metrics"""
        st.header("📊 Campaign Overview")
        
        # Calculate key metrics
        total_people = analysis.get('data_quality', {}).get('total_records', 0)
        barrier_profiles = analysis.get('barrier_profiles', [])
        total_at_risk = sum(p['people_affected'] for p in barrier_profiles)
        
        ml_score = analysis.get('ml_insights', {}).get('model_score', 0)
        data_quality = analysis.get('data_quality', {}).get('completeness_rate', 0)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Population", f"{total_people}")
        
        with col2:
            st.metric("Total At Risk", f"{total_at_risk}", delta=f"{(total_at_risk / total_people * 100):.2f}%")
        
        with col3:
            st.metric("ML Model Accuracy", f"{ml_score:.2%}")
        
        with col4:
            st.metric("Data Completeness", f"{data_quality:.2%}")
        
        with col5:
            st.metric("Unique Barriers Identified", len(barrier_profiles))

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
                color = "#ff4b4b" if priority == "High" else "#ffa44b" if priority == "Medium" else "#00cc88"
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

        barrier_recs = recommendations.get("Barrier Messages", {})
        if not barrier_recs:
            st.warning("No barrier messages available.")
            return

        vaccine_type = st.radio("Select Vaccine Type:", ["H1N1", "Seasonal"], horizontal=True, index=0)
        st.subheader("Message Templates (editable)")

        for idx, (key, details) in enumerate(barrier_recs.items()):
            with st.expander(f"Barrier {idx + 1}: {details.get('insight', '')}"):
                barrier_name = details.get('insight', '').replace('Detected barrier: ', '')
                st.markdown(f"**People affected (approx)**: {details.get('numeric_value', 0)}")

                # Extract messages
                action_text = details.get('action', '')
                h1_msg, s_msg = "", ""
                if "H1N1:" in action_text and "Seasonal:" in action_text:
                    try:
                        h1_msg = action_text.split("H1N1: ")[1].split("\n\nSeasonal: ")[0].strip()
                        s_msg = action_text.split("\n\nSeasonal: ")[1].strip()
                    except:
                        h1_msg = action_text
                        s_msg = action_text
                else:
                    h1_msg = s_msg = action_text

                # Editable fields
                if vaccine_type == "H1N1":
                    h1_msg = st.text_area(f"H1N1 message for '{barrier_name}':", h1_msg, key=f"h1n1_{idx}")
                elif vaccine_type == "Seasonal":
                    s_msg = st.text_area(f"Seasonal message for '{barrier_name}':", s_msg, key=f"seasonal_{idx}")

                # Sample contacts preview
                if 'name' in df.columns and 'phone_number' in df.columns:
                    sample_contacts = df[['name', 'phone_number']].head(3).copy()
                    sample_contacts['h1n1_msg'] = sample_contacts['name'].apply(
                        lambda n: h1_msg.format(name=n) if "{name}" in h1_msg else h1_msg
                    )
                    sample_contacts['seasonal_msg'] = sample_contacts['name'].apply(
                        lambda n: s_msg.format(name=n) if "{name}" in s_msg else s_msg
                    )
                    st.table(sample_contacts)
                else:
                    st.warning("No contact data available (need 'name' and 'phone_number' columns)")

                # Show count for this barrier only
                st.markdown(f"**Messages prepared for this barrier:** {len(details.get('messages_to_send', []))}")

                # Send button for this barrier
                if st.button(f"📤 Send Messages for {barrier_name}", key=f"send_{idx}"):
                    msgs = details.get('messages_to_send', [])
                    if not msgs:
                        st.warning("No messages prepared to send.")
                    else:
                        sent, failed = 0, 0
                        for m in msgs:
                            try:
                                message = client.messages.create(
                                    body=m['h1n1_text'] if vaccine_type == "H1N1" else m['seasonal_text'],
                                    to=m['to'],
                                    from_=from_number
                                )
                                if message.sid:
                                    sent += 1
                                else:
                                    failed += 1
                                    st.error(f"Error sending to {m['to']}: No SID returned.")
                            except Exception as e:
                                failed += 1
                                st.error(f"Error sending to {m['to']}: {str(e)}")
                        st.success(f"✅ Sent: {sent} messages; ❌ Failed: {failed}")
                        st.balloons()

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
        "📊 Overview",
        "📈 Priority Groups",
        "🧠 Top Factors",
        "📨 Messaging",
        "📈 Analysis",
        "📤 Export"
    ])

    with tab1:
        Dashboard.show_overview_metrics(analysis, recommendations)

    with tab2:
        Dashboard.show_priority_groups(recommendations)

    with tab3:
        Dashboard.show_factors(recommendations)

    with tab4:
        Dashboard.show_barrier_messages(recommendations, df)

    with tab5:
        Dashboard.show_analysis_report(analysis, recommendations)

    with tab6:
        Dashboard.setup_export(analysis, recommendations)

if __name__ == "__main__":
    main()