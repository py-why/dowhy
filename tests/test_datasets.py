import numpy as np

from dowhy.datasets import sales_dataset


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
