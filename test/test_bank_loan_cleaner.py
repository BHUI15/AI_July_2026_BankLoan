import pandas as pd
import numpy as np
import pytest
from src.bank_loan_cleaner import BankLoanCleaner

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to provide sample data for testing."""
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

def test_bank_loan_cleaner_preserves_columns(sample_data: pd.DataFrame):
    """Test that the BankLoanCleaner preserves all columns after transformation."""
    cleaner = BankLoanCleaner()
    cleaner.fit(sample_data)
    transformed_data = cleaner.transform(sample_data)

    expect_columns = [
        'person_age', 'person_gender', 'person_income', 'loan_amnt',
        'credit_score', 'previous_loan_defaults_on_file', 'person_education'
    ]

    # Check that all original columns are still present
    assert set(transformed_data.columns) == set(expect_columns), f"Expected columns {expect_columns}, but got {list(transformed_data.columns)}"

def test_bank_loan_cleaner_transform(sample_data: pd.DataFrame):
    """Test the BankLoanCleaner transform method."""
    cleaner = BankLoanCleaner()
    cleaner.fit(sample_data)
    transformed_data = cleaner.transform(sample_data)

    # Check that ages 80 and above are replaced with NaN; Note pandas nan == nan false (use is np.isna() instead)
    assert np.isnan(transformed_data['person_age'].iloc[2])
    assert np.isnan(transformed_data['person_age'].iloc[4])

    # Check that other ages remain unchanged
    assert transformed_data['person_age'].iloc[0] == 25
    assert transformed_data['person_age'].iloc[1] == 30
    assert transformed_data['person_age'].iloc[3] == 45

@pytest.mark.parametrize("column, index, expected_value", [
    ('person_age', 2, np.nan),  # Age 80 should become NaN
    ('person_age', 4, np.nan),  # Age 90 should become NaN
    ('person_age', 0, 25),      # Age 25 should remain unchanged
    ('person_age', 1, 30),      # Age 30 should remain unchanged
    ('person_age', 3, 45),      # Age 45 should remain unchanged
])

def test_bank_loan_cleaner_parametrized(sample_data: pd.DataFrame, column: str, index: int, expected_value):
    """Parametrized test for BankLoanCleaner to check specific values after transformation."""
    cleaner = BankLoanCleaner()
    cleaner.fit(sample_data)
    transformed_data = cleaner.transform(sample_data)

    actual_value = transformed_data[column].iloc[index]
    
    if np.isnan(expected_value):
        assert np.isnan(actual_value), f"Expected NaN at index {index} for column '{column}', but got {actual_value}"
    else:
        assert actual_value == expected_value, f"Expected {expected_value} at index {index} for column '{column}', but got {actual_value}"


def test_bank_loan_cleaner_invalid_input():
    """Test that the BankLoanCleaner raises a ValueError when given invalid input."""
    cleaner = BankLoanCleaner()
    
    # Test with a list instead of a DataFrame
    with pytest.raises(ValueError, match="Expected pandas DataFrame"):
        cleaner.fit([1, 2, 3])
    
    # Test with a string instead of a DataFrame
    with pytest.raises(ValueError, match="Expected pandas DataFrame"):
        cleaner.transform("invalid input")