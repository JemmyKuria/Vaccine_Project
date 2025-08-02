import pandas as pd
import numpy as np

class DataPreprocessor:
    """Class to preprocess survey data for machine learning models"""
    def __init__(self, data, required_columns=None):
        self.data = data.copy()
        self.required_columns = required_columns or [
            'respondent_id', 'h1n1_concern', 'behavioral_antiviral_meds',
            'household_adults', 'household_children', 'doctor_recc_h1n1',
            'doctor_recc_seasonal', 'opinion_h1n1_risk'
        ]
        
    def validate_data(self):
        """Check if required columns exist"""
        missing_cols = [col for col in self.required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return self

    def drop_unused_columns(self):
        """Remove columns with high missingness or irrelevant features"""
        columns_to_drop = [
            'employment_industry', 
            'employment_occupation',
            'hhs_geo_region',
            'census_msa'
        ]
        self.data = self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns])
        return self

    def handle_missing_values(self):
        """Impute missing values based on data type"""
        # Numerical columns: median imputation
        num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].median(), inplace=True)

        # Categorical columns: mode imputation
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        return self

    def engineer_features(self):
        """Create new composite features"""
        # Household features
        if all(col in self.data.columns for col in ['household_adults', 'household_children']):
            self.data['household_size'] = self.data['household_adults'] + self.data['household_children']

        # Doctor recommendations
        if all(col in self.data.columns for col in ['doctor_recc_h1n1', 'doctor_recc_seasonal']):
            self.data['doctor_recc_both'] = (
                self.data['doctor_recc_h1n1'] + self.data['doctor_recc_seasonal']
            )

        # Behavioral score (sum of protective behaviors)
        behavior_cols = [col for col in self.data.columns if 'behavioral_' in col]
        if behavior_cols:
            self.data['safe_behavior_score'] = self.data[behavior_cols].sum(axis=1)

        return self

    def encode_categoricals(self):
        """Convert categorical variables to numerical"""
        # Ordinal encoding
        if 'education' in self.data.columns:
            edu_map = {
                '< 12 Years': 0, 
                '12 Years': 1, 
                'Some College': 2, 
                'College Graduate': 3
            }
            self.data['education'] = self.data['education'].map(edu_map)

        if 'income_poverty' in self.data.columns:
            income_map = {
                'Below Poverty': 0,
                '<= $75,000, Above Poverty': 1,
                '> $75,000': 2
            }
            self.data['income_poverty'] = self.data['income_poverty'].map(income_map)

        # One-hot encoding
        ohe_cols = ['race', 'sex', 'marital_status', 'rent_or_own']
        for col in ohe_cols:
            if col in self.data.columns:
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data.drop(columns=[col], inplace=True)

        return self

    def split_features_labels(self):
        """Separate features from labels (only for training data)"""
        if self.is_training_data:
            self.labels = self.data[self.label_columns]
            self.features = self.data.drop(columns=self.label_columns + ['respondent_id'])
        else:
            self.features = self.data.drop(columns=['respondent_id'])
        return self


    def preprocess(self):
        """Run full preprocessing pipeline"""
        return (
            self.validate_data()
            .drop_unused_columns()
            .handle_missing_values()
            .engineer_features()
            .encode_categoricals()
            .split_features_labels()
        )

    def get_processed_data(self):
        """Return cleaned DataFrame"""
        return self.data


# Example usage (for testing)
if __name__ == "__main__":
    # Test with sample data
    raw_data = pd.read_csv("data/raw/training_set_features.csv")
    preprocessor = DataPreprocessor(raw_data)
    processed_data = preprocessor.preprocess().get_processed_data()
    print(f"Processed data shape: {processed_data.shape}")
    print("Sample columns:", processed_data.columns.tolist()[:10])