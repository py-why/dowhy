import numpy as np

from dowhy.datasets import linear_dataset, sales_dataset


def test_when_generating_sales_dataset_then_returns_reasonable_samples():
    sales_df1 = sales_dataset("2021-01-01", "2021-12-31", num_shopping_events=10)
    sales_df2 = sales_dataset(
        "2021-01-01", "2021-12-31", num_shopping_events=10, change_of_price=0.9, page_visitor_factor=0.85
    )

    assert np.sum(sales_df1["Shopping Event?"]) == 10
    assert np.sum(sales_df2["Shopping Event?"]) == 10
    assert np.mean(sales_df1["Profit"]) > np.mean(sales_df2["Profit"])
    assert np.mean(sales_df1["Page Views"]) > np.mean(sales_df2["Page Views"])
    assert np.mean(sales_df1["Sold Units"]) < np.mean(sales_df2["Sold Units"])


def test_linear_dataset_with_binary_treatment():
    """Regression test for gh-1388: treatment_is_binary fails on NumPy >= 2.4."""
    data = linear_dataset(
        beta=10,
        num_common_causes=5,
        num_instruments=2,
        num_samples=100,
        treatment_is_binary=True,
    )
    df = data["df"]
    treatment_col = data["treatment_name"][0]
    assert set(df[treatment_col].unique()).issubset({0, 1})


def test_linear_dataset_with_categorical_treatment():
    """Regression test for gh-1388: treatment_is_category uses the same pattern."""
    data = linear_dataset(
        beta=10,
        num_common_causes=3,
        num_samples=100,
        treatment_is_binary=False,
        treatment_is_category=True,
    )
    df = data["df"]
    treatment_col = data["treatment_name"][0]
    assert set(df[treatment_col].unique()).issubset({0, 1, 2})
