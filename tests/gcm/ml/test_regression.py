import numpy as np
from _pytest.python_api import approx

from dowhy.gcm.ml.regression import create_product_regressor


def test_when_use_product_regressor_then_computes_correct_values():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mdl = create_product_regressor()
    # No fit needed

    assert mdl.predict(X).reshape(-1) == approx(np.array([6, 120, 504]))


def test_when_input_is_categorical_when_use_product_regressor_then_computes_correct_values():
    X = np.column_stack([np.array(["Class 1", "Class 2"]).astype(object), np.array([1, 2])]).astype(object)

    mdl = create_product_regressor()
    mdl.fit(X, np.zeros(2))  # Need to fit one-hot-encoder

    assert mdl.predict(X).reshape(-1) == approx(np.array([0, 2]))
