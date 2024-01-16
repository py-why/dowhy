import networkx as nx
import numpy as np
import pandas as pd
from _pytest.python_api import approx

from dowhy.utils.encoding import one_hot_encode


def test_one_hot_encode_equivalent_to_get_dummies():

    # Use a mix of already-numeric and requires encoding cols:
    data = {
        "C": ["X", "Y", "Z", "X", "Y", "Z"],
        "N": [1, 2, 3, 4, 5, 6],
    }
    df = pd.DataFrame(data)

    # NB There may be small differences in type but since all values will be used in models as float,
    # comparison is done as this type.
    df_dummies = pd.get_dummies(df, drop_first=True)
    df_dummies = df_dummies.astype(float)

    df_sklearn, _ = one_hot_encode(df, drop_first=True)
    df_sklearn = df_sklearn.astype(float)

    # Check same rows
    len1 = len(df_dummies)
    len2 = len(df_sklearn)
    assert len1 == len2

    # Check same number of cols
    len1 = len(df_dummies.columns)
    len2 = len(df_sklearn.columns)
    assert len1 == len2

    # Check values
    # Calculate the sum of absolute differences between the two DataFrames
    # - should be zero (excl. floating point error)
    sum_abs_diff = (df_dummies - df_sklearn).abs().sum().sum()
    assert sum_abs_diff == approx(0.0)


def test_one_hot_encode_consistent_with_new_data():

    # Use a mix of already-numeric and requires encoding cols:
    data1 = {
        "C": ["X", "Y", "Z", "X", "Y", "Z"],
        "N": [1, 2, 3, 4, 5, 6],
    }
    df1 = pd.DataFrame(data1)

    # Initial encode
    df_encoded1, encoder = one_hot_encode(df1, drop_first=True)
    df_encoded1 = df_encoded1.astype(float)

    # Create new data with permuted rows.
    # Output shape should be unchanged.
    data2 = {
        "C": ["Y", "Z", "X", "X", "Y", "Z"],
        "N": [1, 2, 3, 4, 5, 6],
    }
    df2 = pd.DataFrame(data2)

    # Encode this new data.
    df_encoded2, _ = one_hot_encode(df2, encoder=encoder, drop_first=True)
    df_encoded2 = df_encoded2.astype(float)

    # Check same rows
    len1 = len(df_encoded1)
    len2 = len(df_encoded2)
    assert len1 == len2

    # Check same number of cols
    len1 = len(df_encoded1.columns)
    len2 = len(df_encoded2.columns)
    assert len1 == len2

    # Check permuted values are consistent
    c_y1 = df_encoded1["C_Y"]
    c_y2 = df_encoded2["C_Y"]
    assert c_y1[1] == c_y2[0]
    assert c_y1[4] == c_y2[4]

    c_z1 = df_encoded1["C_Z"]
    c_z2 = df_encoded2["C_Z"]
    assert c_z1[2] == c_z2[1]
    assert c_z1[5] == c_z2[5]
