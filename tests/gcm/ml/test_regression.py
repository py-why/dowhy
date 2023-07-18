import numpy as np
from _pytest.python_api import approx
from flaky import flaky

from dowhy.gcm.ml.regression import create_polynom_regressor, create_random_forest_regressor


@flaky(max_runs=3)
def test_when_fit_and_predict_polynom_regressor_then_returns_accurate_results():
    X = np.random.normal(0, 1, (100, 2))
    Y = X[:, 0] * X[:, 1]

    mdl = create_polynom_regressor(degree=2)
    mdl.fit(X, Y)

    X_test = np.random.normal(0, 1, (100, 2))
    Y_test = X_test[:, 0] * X_test[:, 1]

    assert mdl.predict(X_test).reshape(-1) == approx(Y_test, abs=1e-10)


@flaky(max_runs=3)
def test_given_categorical_training_data_when_fit_and_predict_polynom_regressor_then_returns_accurate_results():
    def _generate_data():
        X = np.column_stack(
            [np.random.choice(2, 100, replace=True).astype(str), np.random.normal(0, 1, (100, 2)).astype(object)]
        ).astype(object)
        Y = []
        for i in range(X.shape[0]):
            Y.append(X[i, 1] * X[i, 2] if X[i, 0] == "0" else X[i, 1] + X[i, 2])

        return X, np.array(Y)

    X_training, Y_training = _generate_data()
    X_test, Y_test = _generate_data()
    mdl = create_polynom_regressor(degree=3)
    mdl.fit(X_training, Y_training)

    assert mdl.predict(X_test).reshape(-1) == approx(Y_test, abs=1e-10)


@flaky(max_runs=3)
def test_given_categorical_training_data_with_many_categories_when_fit_regression_model_then_returns_reasonably_accurate_predictions():
    def _generate_data():
        X = np.column_stack(
            [np.random.choice(20, 10000, replace=True).astype(str), np.random.normal(0, 1, (10000, 2)).astype(object)]
        ).astype(object)
        Y = [X[i, 1] * X[i, 1 + int(X[i, 0]) % 2] for i in range(X.shape[0])]

        return X, np.array(Y)

    X_training, Y_training = _generate_data()
    X_test, Y_test = _generate_data()
    mdl = create_random_forest_regressor()
    mdl.fit(X_training, Y_training)

    assert np.mean((mdl.predict(X_test).reshape(-1) - Y_test) ** 2) < 0.1
