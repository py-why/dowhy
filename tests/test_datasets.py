import numpy as np

from dowhy.datasets import convert_to_categorical, create_dot_graph, create_gml_graph, linear_dataset, sales_dataset


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


def test_convert_to_categorical_default_quantiles_not_shared():
    """Ensure mutable default quantiles are not shared across calls (regression for B006)."""
    arr = np.random.randn(50, 3)
    # Two calls with default quantiles should produce identical results
    result1 = convert_to_categorical(arr, num_vars=3, num_discrete_vars=1)
    result2 = convert_to_categorical(arr, num_vars=3, num_discrete_vars=1)
    np.testing.assert_array_equal(result1, result2)


def test_convert_to_categorical_custom_quantiles():
    """Custom quantiles are respected and the default list is not mutated."""
    arr = np.random.randn(50, 2)
    custom_q = [0.33, 0.66]
    result = convert_to_categorical(arr, num_vars=2, num_discrete_vars=1, quantiles=custom_q)
    # With 2 quantile cuts there should be 3 unique categories (0, 1, 2)
    assert set(result[:, -1]).issubset({0, 1, 2, 3})
    # Caller's list must not have been mutated
    assert custom_q == [0.33, 0.66]


def test_create_dot_graph_no_optional_args():
    """create_dot_graph works without optional effect_modifiers or frontdoor_variables."""
    dot = create_dot_graph(["T"], "Y", ["W"], ["Z"])
    assert "T->Y" in dot
    assert "W->T" in dot
    assert "Z->T" in dot


def test_create_dot_graph_with_effect_modifiers():
    """effect_modifiers are included in the graph."""
    dot = create_dot_graph(["T"], "Y", ["W"], [], effect_modifiers=["M"])
    assert "M->Y" in dot


def test_create_gml_graph_no_optional_args():
    """create_gml_graph works without optional effect_modifiers or frontdoor_variables."""
    gml = create_gml_graph(["T"], "Y", ["W"], ["Z"])
    assert '"T"' in gml
    assert '"Y"' in gml
    assert '"W"' in gml
