# Flu Shot Learning: Vaccine Uptake Prediction

## Project Overview
This project predicts the likelihood of individuals receiving the seasonal flu vaccine, the H1N1 vaccine, or both, using demographic, behavioural, and opinion-based survey data from the DrivenData Flu Shot Learning challenge.  
The ultimate goal is to generate actionable insights and create a tool that supports targeted public health interventions.

## Problem Statement
Despite the availability of vaccines and public health campaigns, a significant portion of the population remains unvaccinated, leaving communities vulnerable to outbreaks.  
During pandemics such as H1N1 in 2009, identifying hesitant groups can be critical in directing resources and messaging.  
This project uses survey data to predict vaccination uptake and support smarter, targeted public health strategies.

## Objectives
- Identify the most significant factors influencing vaccine uptake.
- Compare multiple classification algorithms to determine the most effective predictive model.
- Deploy the best-performing model as a user-friendly web application.

## Data Understanding
**Source:** DrivenData Flu Shot Learning Challenge  
**Size:** ~26,000 records (train + test + labels)  
**Features:**  
- Demographics (age group, gender, income level, education, marital status)  
- Health & Access (chronic illness, health insurance, doctor visits)  
- Beliefs (perceived flu risk, trust in vaccines, side effect concerns)  
- Behaviour (following doctor's advice, reading flu-related news)  

**Target Variables:**  
- `h1n1_vaccine`: 1 if received H1N1 vaccine, else 0  
- `seasonal_vaccine`: 1 if received seasonal flu vaccine, else 0  

## Exploratory Data Analysis (EDA)
EDA was performed to:
- Examine feature distributions and missing values.
- Identify correlations between features and vaccine uptake.
- Understand demographic patterns in vaccination behaviour.

## Basic Modelling
Several classification models were tested, including:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Other ensemble methods

Models were evaluated on precision, recall, F1-score, and ROC-AUC.

## Hyperparameter Tuning
Hyperparameter tuning was applied to the most promising models to improve predictive performance.  
Gradient Boosting emerged as the top-performing model after tuning, balancing precision, recall, and overall accuracy.

## Deployment
The Gradient Boosting model was deployed as a Streamlit web application:  
🔗 **[VaxTrend Web App](https://vaxtrend.streamlit.app/)**  
The app allows users to input individual characteristics and receive probability predictions for both vaccines.

## Conclusion
The project demonstrated the feasibility of predicting vaccine uptake using survey data.  
The deployed model and app provide a practical tool for public health professionals to target campaigns and resources effectively.

## Next Steps
- Expand dataset to include other regions for improved generalisability.
- Conduct bias and fairness analysis.
- Gather user feedback to refine the application.