import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BankLoanCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        #Validation: ensure input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(X)}")

        #Store input features during fit so sklearn knows about them
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        #Validation: ensure input is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(X)}")

        X_cleaned = X.copy()
        
        if 'person_age' in X_cleaned.columns:
            # Replace ages 80 and above with NaN
            X_cleaned['person_age'] = X_cleaned['person_age'].mask(X_cleaned['person_age'] >= 80, np.nan) 

        return X_cleaned

    def get_feature_names_out(self, input_features=None):
        # We don't drop or add columns here, so it's a 1:1 pass-through
        if input_features is None:
            return self.feature_names_in_

        return np.array(input_features)
