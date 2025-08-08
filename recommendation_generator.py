import pandas as pd

def generate_recommendation_summary(df: pd.DataFrame, target_column='h1n1_vaccine_pred') -> pd.DataFrame:
    """
    Takes a DataFrame with predictions and returns a summary of barrier profiles + recommended actions.
    """

    # Filter to individuals predicted not to take the vaccine
    non_takers = df[df[target_column] == 0].copy()

    # Generate barrier flags
    non_takers['no_doc_rec'] = non_takers['doctor_recc_both'] == 0
    non_takers['low_vaccine_belief'] = non_takers['opinion_h1n1_vacc_effective'] <= 2
    non_takers['low_risk_perception'] = non_takers['opinion_h1n1_risk'] <= 2
    non_takers['low_knowledge'] = non_takers['h1n1_knowledge'] <= 1
    non_takers['no_insurance'] = non_takers['health_insurance'] == 0
    non_takers['low_safe_behavior'] = non_takers['safe_behavior_score'] <= 2

    # Create barrier profile string
    def create_profile(row):
        profile = []
        if row['no_doc_rec']: profile.append("No Doctor Rec")
        if row['low_vaccine_belief']: profile.append("Low Vaccine Belief")
        if row['low_risk_perception']: profile.append("Low Risk Perception")
        if row['low_knowledge']: profile.append("Low Knowledge")
        if row['no_insurance']: profile.append("No Insurance")
        if row['low_safe_behavior']: profile.append("Low Safe Behavior")
        return " + ".join(profile) if profile else "No Major Barriers"

    non_takers['barrier_profile'] = non_takers.apply(create_profile, axis=1)

    # Count how many people fall into each profile
    profile_counts = non_takers['barrier_profile'].value_counts().reset_index()
    profile_counts.columns = ['barrier_profile', 'people_affected']

    # Define a mapping from barrier to recommendation
    recommendation_map = {
        'No Doctor Rec + Low Vaccine Belief': "Use CHVs and mobile clinics.",
        'Low Knowledge + No Insurance': "Send SMS reminders and promote free access.",
        'Low Risk Perception + Low Vaccine Belief': "Mass media campaigns highlighting flu dangers.",
        'Low Safe Behavior + Low Knowledge': "Distribute hygiene & flu education materials.",
        'No Insurance': "Partner with community groups to lower financial barriers.",
        'Low Vaccine Belief': "Use trusted leaders to spread vaccine facts.",
        'No Major Barriers': "General encouragement via SMS or posters."
        # Add more mappings as needed
    }

    def map_recommendation(profile):
        for key in recommendation_map:
            if key in profile:
                return recommendation_map[key]
        return "General outreach: education, access, and trust-building."

    profile_counts['recommended_action'] = profile_counts['barrier_profile'].apply(map_recommendation)

    return profile_counts

    
def export_recommendations(recommendations: pd.DataFrame, filename='recommendations.csv'):
    """
    Exports the recommendations DataFrame to a CSV file.
    """
    recommendations.to_csv(filename, index=False)
    print(f"Recommendations exported to {filename}")
def display_key_insights(h1n1_non_takers: pd.DataFrame, seasonal_non_takers: pd.DataFrame):
    """
    Displays key insights about the non-takers of H1N1 and seasonal vaccines.
    """
    st.subheader("Key Insights")
    
    h1n1_count = len(h1n1_non_takers)
    seasonal_count = len(seasonal_non_takers)
    
    st.write(f"**H1N1 Non-Takers:** {h1n1_count} individuals")
    st.write(f"**Seasonal Non-Takers:** {seasonal_count} individuals")
    
    if h1n1_count > 0:
        st.write("### H1N1 Vaccine Non-Takers")
        st.dataframe(h1n1_non_takers.head())
    
    if seasonal_count > 0:
        st.write("### Seasonal Vaccine Non-Takers")
        st.dataframe(seasonal_non_takers.head())
def display_targeted_interventions(h1n1_recommendations: pd.DataFrame, seasonal_recommendations: pd.DataFrame):
    """
    Displays targeted interventions based on barrier profiles.
    """
    st.subheader("Targeted Interventions")
    
    if not h1n1_recommendations.empty:
        st.write("### H1N1 Vaccine Recommendations")
        st.dataframe(h1n1_recommendations)
    
    if not seasonal_recommendations.empty:
        st.write("### Seasonal Vaccine Recommendations")
        st.dataframe(seasonal_recommendations)
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Vaccine Hesitancy Recommendations")

    # Load the results DataFrame from session state
    if 'results_df' not in st.session_state:
        st.error("No results found. Please process data first.")
        return

    results = st.session_state['results_df']

    # Filter non-takers
    h1n1_non_takers = results[results['h1n1_label'] == 0]
    seasonal_non_takers = results[results['seasonal_label'] == 0]

    # Generate recommendations
    h1n1_recommendations = generate_recommendation_summary(h1n1_non_takers, target_column='h1n1_label')
    seasonal_recommendations = generate_recommendation_summary(seasonal_non_takers, target_column='seasonal_label')

    # Display key insights
    display_key_insights(h1n1_non_takers, seasonal_non_takers)

    # Display targeted interventions
    display_targeted_interventions(h1n1_recommendations, seasonal_recommendations)

    # Export recommendations
    export_recommendations(h1n1_recommendations, filename='h1n1_recommendations.csv')
    export_recommendations(seasonal_recommendations, filename='seasonal_recommendations.csv')