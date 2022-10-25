import numpy as np
from _pytest.python_api import approx

from dowhy.gcm.ml.regression import create_product_regressor


def test_given_product_regressor_then_computes_correct_values():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mdl = create_product_regressor()

    assert mdl.predict(X).reshape(-1) == approx(np.array([6, 120, 504]))
