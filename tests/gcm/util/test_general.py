import numpy as np
import pandas as pd

from dowhy.gcm.util.general import has_categorical, is_categorical


def test_given_categorical_data_when_evaluating_is_categorical_then_returns_expected_result():
    assert is_categorical(np.array(["A", "B", "A"]))
    assert is_categorical(np.array([True, False, False]))
    assert is_categorical(pd.DataFrame({"X": [True, False, False]}).to_numpy())
    assert not is_categorical(np.array([1, 2, 3]))


def test_given_categorical_data_when_evaluating_has_categorical_then_returns_expected_result():
    assert has_categorical(np.array([["A", 2, 3], ["B", 4, 5]]))
    assert has_categorical(
        np.column_stack([np.array([True, False, False], dtype=object), np.array([1, 2, 3], dtype=object)])
    )
    assert has_categorical(pd.DataFrame({"X": [True, False, False], "Y": [1, 2, 3]}).to_numpy())
    assert not has_categorical(np.array([[1, 2, 3], [12.2, 2.3, 3.231]]))
