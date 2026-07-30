import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_person_income=True):
        self.drop_person_income = drop_person_income

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # Example feature engineering: create a new feature by multiplying two existing features
        if 'person_income' in X_transformed.columns:
            X_transformed["logged_person_income"] = np.log(X_transformed["person_income"] + 1)  # Adding 1 to avoid log(0)
            X_transformed = X_transformed.drop(columns=["person_income"]) if self.drop_person_income else X_transformed

        # Drop unrelated features
        X_transformed = X_transformed[['person_age', 'person_gender', 'logged_person_income','loan_amnt',
       'credit_score', 'previous_loan_defaults_on_file']]

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # We must explicitily tell sklearn which columns we dropped or added
        features =(
            list(input_features) if input_features is not None else list(self.feature_names_in_)
        )
        
        if "person_income" in features and self.drop_person_income:
            item_tobe_removed = ['person_education', 'person_income',
                            'person_emp_exp', 'person_home_ownership', 'loan_intent',
                            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
            for item in item_tobe_removed:
                if item in features:
                    features.remove(item)
            features.append("logged_person_income")

        return np.array(features, dtype=object)