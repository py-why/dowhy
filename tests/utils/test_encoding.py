import pandas as pd
import pytest
from _pytest.python_api import approx

from dowhy.utils.encoding import Encoders, one_hot_encode


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


def test_one_hot_encode_explicit_columns():
    df = pd.DataFrame({"A": ["x", "y", "x"], "B": ["p", "q", "p"], "N": [1, 2, 3]})
    result, _ = one_hot_encode(df, columns=["A"], drop_first=False)
    assert "A_x" in result.columns
    assert "A_y" in result.columns
    # B should be untouched (not in columns list)
    assert "B" in result.columns
    assert "N" in result.columns


def test_one_hot_encode_non_list_columns_raises():
    df = pd.DataFrame({"C": ["x", "y"]})
    with pytest.raises(TypeError):
        one_hot_encode(df, columns="C")


def test_one_hot_encode_all_numeric_returns_unchanged():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4, 5, 6]})
    result, encoder = one_hot_encode(df)
    assert list(result.columns) == list(df.columns)
    assert encoder is None


class TestEncoders:
    def _make_df(self):
        return pd.DataFrame({"C": ["x", "y", "z", "x"], "N": [1, 2, 3, 4]})

    def test_encode_produces_one_hot_columns(self):
        encoders = Encoders()
        df = self._make_df()
        result = encoders.encode(df, "train")
        assert "C_x" in result.columns or "C_y" in result.columns

    def test_encode_reuses_encoder_on_second_call(self):
        encoders = Encoders()
        df = self._make_df()
        result1 = encoders.encode(df, "train")
        result2 = encoders.encode(df, "train")
        assert list(result1.columns) == list(result2.columns)

    def test_encode_creates_separate_encoders_per_name(self):
        encoders = Encoders()
        df1 = pd.DataFrame({"A": ["a", "b"]})
        df2 = pd.DataFrame({"B": ["p", "q"]})
        r1 = encoders.encode(df1, "enc1")
        r2 = encoders.encode(df2, "enc2")
        assert "A_b" in r1.columns or "A_a" in r1.columns
        assert "B_q" in r2.columns or "B_p" in r2.columns

    def test_reset_clears_encoders(self):
        encoders = Encoders()
        df = self._make_df()
        encoders.encode(df, "train")
        assert "train" in encoders._encoders

        encoders.reset()
        assert len(encoders._encoders) == 0

    def test_encode_after_reset_creates_new_encoder(self):
        encoders = Encoders()
        df = self._make_df()
        encoders.encode(df, "train")
        encoders.reset()
        result = encoders.encode(df, "train")
        # Result should still be valid — encoder was re-created
        assert len(result) == len(df)

    def test_drop_first_true(self):
        encoders = Encoders(drop_first=True)
        df = pd.DataFrame({"C": ["x", "y", "z"]})
        result = encoders.encode(df, "train")
        # drop_first=True means k-1 columns for k categories
        encoded_cols = [c for c in result.columns if c.startswith("C_")]
        assert len(encoded_cols) == 2

    def test_drop_first_false(self):
        encoders = Encoders(drop_first=False)
        df = pd.DataFrame({"C": ["x", "y", "z"]})
        result = encoders.encode(df, "train")
        encoded_cols = [c for c in result.columns if c.startswith("C_")]
        assert len(encoded_cols) == 3

    def test_encode_consistency_across_new_data(self):
        encoders = Encoders()
        df_train = pd.DataFrame({"C": ["x", "y", "z"], "N": [1, 2, 3]})
        df_test = pd.DataFrame({"C": ["z", "x", "y"], "N": [4, 5, 6]})
        train_result = encoders.encode(df_train, "data")
        test_result = encoders.encode(df_test, "data")
        # Same columns, same shape
        assert list(train_result.columns) == list(test_result.columns)
        assert len(train_result) == len(df_train)
        assert len(test_result) == len(df_test)
