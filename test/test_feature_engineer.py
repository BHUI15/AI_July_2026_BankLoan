import pandas as pd
import numpy as np
import pytest
from src.feature_engineer import FeatureEngineer

@pytest.fixture
def raw_data() -> pd.DataFrame:
    """Fixture to provide raw data for testing."""
    data = {
        'person_age': [25, 30, 80, 45, 90],
        'person_gender': ['male', 'female', 'male', 'female', 'male'],
        'person_income': [50000, 60000, 70000, 80000, 90000],
        'loan_amnt': [10000, 15000, 20000, 25000, 30000],
        'credit_score': [700, 750, 800, 650, 600],
        'previous_loan_defaults_on_file': ['No', 'Yes', 'No', 'Yes', 'No'],
        'person_education': ['Bachelor', 'Master', 'Doctorate', 'High School', 'Associate'],
    }
    return pd.DataFrame(data)

def test_feature_engineer_transform(raw_data: pd.DataFrame):
    """Test the FeatureEngineer transform method."""
    engineer = FeatureEngineer(drop_person_income=True)
    engineer.fit(raw_data)
    transformed_data = engineer.transform(raw_data)

    # Check that 'person_income' is dropped and 'logged_person_income' is added
    assert 'person_income' not in transformed_data.columns
    assert 'logged_person_income' in transformed_data.columns

    # Check that the logged values are correct
    expected_logged_values = (raw_data['person_income'] + 1).apply(np.log)
    pd.testing.assert_series_equal(transformed_data['logged_person_income'], expected_logged_values, check_names=False)

    # Check that unrelated features are dropped
    expected_columns = ['person_age', 'person_gender', 'logged_person_income', 'loan_amnt', 'credit_score', 'previous_loan_defaults_on_file']
    assert set(transformed_data.columns) == set(expected_columns), f"Expected columns {expected_columns}, but got {list(transformed_data.columns)}" 

