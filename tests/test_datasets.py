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


def test_linear_dataset_return_all_coefficients():
    """return_all_coefficients=True exposes ground-truth beta, c1, c2, etc."""
    data = linear_dataset(
        beta=5,
        num_common_causes=2,
        num_instruments=1,
        num_effect_modifiers=1,
        num_samples=200,
        return_all_coefficients=True,
    )
    assert "beta" in data
    assert "c1" in data
    assert "c2" in data
    assert "ce" in data
    assert "cz" in data
    # cfd1/cfd2 are None when num_frontdoor_variables=0
    assert data["cfd1"] is None
    assert data["cfd2"] is None
    # beta matches the requested value
    assert np.allclose(data["beta"], [5])
    # c1: (num_W_cols, num_treatments), c2: (num_W_cols,)
    assert data["c1"].shape[1] == 1
    assert data["c2"].shape[0] == data["c1"].shape[0]


def test_linear_dataset_return_all_coefficients_false_by_default():
    """Coefficients are NOT included in the dict by default."""
    data = linear_dataset(beta=1, num_common_causes=1, num_samples=100)
    assert "beta" not in data
    assert "c1" not in data
    assert "c2" not in data


def test_linear_dataset_return_all_coefficients_with_frontdoor():
    """Frontdoor coefficients are populated when num_frontdoor_variables > 0."""
    data = linear_dataset(
        beta=3,
        num_common_causes=1,
        num_frontdoor_variables=2,
        num_samples=200,
        return_all_coefficients=True,
    )
    assert data["cfd1"] is not None
    assert data["cfd2"] is not None
    assert data["cfd1"].shape == (1, 2)  # (num_treatments, num_frontdoor_variables)
    assert data["cfd2"].shape == (2,)  # (num_frontdoor_variables,)


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
