# pipeline.py
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder


import os
import gdown

# Only download if the file isn't already there
if not os.path.exists("multi_tuned_rf.pkl"):
    print("Downloading model from Google Drive...")
    gdown.download("https://drive.google.com/file/d/1Kw0k7CTZNNOypwYuyjvF02WDOWs7U2I4/view?usp=drive_link", "multi_tuned_rf.pkl", quiet=False)


# paths
MODEL_PATH = "multi_tuned_rf.pkl" 

# load once
model = joblib.load(MODEL_PATH)
print("Model loaded:", model)

EXPECTED_COLS = [
    'h1n1_concern','h1n1_knowledge','behavioral_antiviral_meds',
    'chronic_med_condition','child_under_6_months','health_worker',
    'health_insurance','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective',
    'opinion_seas_risk','opinion_seas_sick_from_vacc','age_group',
    'education','income_poverty','household_size','doctor_recc_both',
    'safe_behavior_score','race_Hispanic','race_Other or Multiple',
    'race_White','sex_Male','marital_status_Not Married',
    'rent_or_own_Rent','employment_status_Not in Labor Force',
    'employment_status_Unemployed'
]  



# 1. Pre-processing 

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # drop identifier & targets if they exist
    for col in ["respondent_id", "h1n1_vaccine", "seasonal_vaccine"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # handle high-missing employment columns 
    for col in ["employment_industry", "employment_occupation","hhs_geo_region", "census_msa"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # fill health_insurance
    df["health_insurance"] = df["health_insurance"].fillna("NA")
    df["health_insurance"] = df["health_insurance"].map({1.0: 1, 0.0: 0}).fillna(-1)

    # ordinal maps
    edu_order = {"< 12 Years": 0, "12 Years": 1, "Some College": 2, "College Graduate": 3}
    inc_order = {"Below Poverty": 0, "<= $75,000, Above Poverty": 1, "> $75,000": 2}
    age_order = {"18 - 34 Years": 0, "35 - 44 Years": 1, "45 - 54 Years": 2,
                 "55 - 64 Years": 3, "65+ Years": 4}

    for col, mapper in {"education": edu_order, "income_poverty": inc_order, "age_group": age_order}.items():
        if col in df.columns:
            df[col] = df[col].map(mapper)

    # engineered features
    df["household_size"] = df["household_adults"].fillna(0) + df["household_children"].fillna(0)

    safe_behaviors = [
        "behavioral_avoidance", "behavioral_face_mask", "behavioral_wash_hands",
        "behavioral_large_gatherings", "behavioral_outside_home", "behavioral_touch_face"
    ]
    df["safe_behavior_score"] = df[safe_behaviors].sum(axis=1)

    # doctor_recc_both
    df["doctor_recc_both"] = df["doctor_recc_h1n1"].fillna(0) + df["doctor_recc_seasonal"].fillna(0)
    df.drop(columns=["doctor_recc_h1n1", "doctor_recc_seasonal"], inplace=True)

    # one-hot nominal columns
    cats = ["race", "sex", "marital_status", "rent_or_own", "employment_status"]
    df = pd.get_dummies(df, columns=cats, drop_first=True)

    # numeric impute
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # ensuring columns match model expectation
    for col in EXPECTED_COLS:
        if col not in df:
            df[col] = 0
    df = df[EXPECTED_COLS]

    return df


# 2. Predict
def predict(df_processed: pd.DataFrame):
    labels = model.predict(df_processed)          # returns a 2-column array
    h1n1_label    = pd.Series(labels[:, 0], index=df_processed.index)
    seasonal_label= pd.Series(labels[:, 1], index=df_processed.index)
    return h1n1_label, seasonal_label


    