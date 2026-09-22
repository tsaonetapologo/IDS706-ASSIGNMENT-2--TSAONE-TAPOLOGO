import pandas as pd
import pytest

from ecommerce_analysis import clean_data, get_high_spenders, average_spending_by_age


@pytest.fixture
def sample_data():
    """Create a small dataset for testing."""
    return pd.DataFrame(
        {
            "Age": [20, 30, 20, 40],
            "Purchase_Amount": ["$500", "$1,500", "$2,000", "$800"],
            "Frequency_of_Purchase": [2, 4, 3, 1],
            "Time_to_Decision": [5, 10, 7, 3],
            "Gender": ["Female", None, "Male", "Female"],
        }
    )


def test_clean_data_converts_purchase_amount(sample_data):
    result = clean_data(sample_data)

    assert pd.api.types.is_numeric_dtype(result["Purchase_Amount"])

    assert result["Purchase_Amount"].tolist() == [500, 1500, 2000, 800]


def test_clean_data_fills_missing_categories(sample_data):
    result = clean_data(sample_data)

    assert result["Gender"].isna().sum() == 0
    assert "Unknown" in result["Gender"].values


def test_clean_data_removes_missing_required_values():
    df = pd.DataFrame(
        {
            "Age": [20, None, 30],
            "Purchase_Amount": [500, 700, None],
            "Frequency_of_Purchase": [2, 3, 4],
            "Time_to_Decision": [5, 6, 7],
        }
    )

    result = clean_data(df)

    assert len(result) == 1
    assert result.iloc[0]["Age"] == 20


def test_get_high_spenders(sample_data):
    df = clean_data(sample_data)

    result = get_high_spenders(df)

    assert len(result) == 2
    assert all(result["Purchase_Amount"] > 1000)


def test_high_spenders_custom_threshold(sample_data):
    df = clean_data(sample_data)

    result = get_high_spenders(df, threshold=700)

    assert len(result) == 3


def test_average_spending_by_age(sample_data):
    df = clean_data(sample_data)

    result = average_spending_by_age(df)

    assert result.loc[20] == 1250
    assert result.loc[30] == 1500
    assert result.loc[40] == 800


def test_high_spenders_excludes_amount_equal_to_threshold(sample_data):
    df = clean_data(sample_data)

    result = get_high_spenders(df, threshold=1000)

    assert not any(result["Purchase_Amount"] == 1000)


def test_high_spenders_returns_empty_when_none_qualify(sample_data):
    df = clean_data(sample_data)

    result = get_high_spenders(df, threshold=10000)

    assert result.empty
