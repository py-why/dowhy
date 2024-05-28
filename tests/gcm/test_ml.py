import numpy as np
import pytest
from flaky import flaky
from pytest import approx
from sklearn.linear_model import LogisticRegression

from dowhy.gcm.ml import (
    SklearnClassificationModel,
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_logistic_regression_classifier,
)


@flaky(max_runs=5)
def test_given_categorical_input_feature_when_fit_regression_model_then_use_it_correctly():
    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)
    X2 = []

    for i in range(1000):
        tmp_value = 2 * X0[i]

        if X1[i] == "0":
            tmp_value -= 5
        elif X1[i] == "1":
            tmp_value += 10
        else:
            tmp_value += 5

        X2.append(tmp_value)

    X2 = np.array(X2)

    inputs = np.array([X0, X1], dtype=object).T

    mdl = create_linear_regressor()
    mdl.fit(inputs, X2)

    assert mdl.predict(np.array([[2, "1"]], dtype=object)) == approx(14)
    assert mdl.predict(inputs) == approx(X2.reshape(-1, 1))


def test_given_categorical_input_and_output_features_when_fit_classification_model_then_use_it_correctly():
    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)
    X2 = []

    for i in range(1000):
        tmp_value = 2 * X0[i]

        if X1[i] == "0":
            tmp_value -= 5
        elif X1[i] == "1":
            tmp_value += 10
        else:
            tmp_value += 5

        X2.append(tmp_value)

    X2 = (X2 > np.median(X2)).astype(str)

    inputs = np.array([X0, X1], dtype=object).T

    mdl = create_logistic_regression_classifier()
    mdl.fit(inputs, X2)

    assert mdl.predict_probabilities(np.array([[2, "1"]], dtype=object)) == approx(np.array([[0, 1]]), abs=0.01)
    assert np.sum(mdl.predict(inputs).reshape(-1) != X2) < 20

    _, counts = np.unique(mdl.predict(inputs), return_counts=True)
    assert counts / 1000 == approx(np.array([0.5, 0.5]), abs=0.05)


def test_when_cloning_sklearn_classification_model_then_returns_a_cloned_object():
    logistic_regression_model = LogisticRegression()
    mdl = SklearnClassificationModel(logistic_regression_model)
    cloned_mdl = mdl.clone()

    assert isinstance(cloned_mdl, SklearnClassificationModel)
    assert isinstance(cloned_mdl.sklearn_model, LogisticRegression)
    assert mdl != cloned_mdl
    assert cloned_mdl.sklearn_model != logistic_regression_model


def test_when_using_linear_regressor_with_given_parameters_then_fit_does_not_override_parameters():
    mdl = create_linear_regressor_with_given_parameters([1, 2, 3], 4)

    assert mdl.coefficients == pytest.approx([1, 2, 3])
    assert mdl.intercept == 4

    mdl.fit(np.random.normal(0, 1, (100, 3)), np.arange(100))

    assert mdl.coefficients == pytest.approx([1, 2, 3])
    assert mdl.intercept == 4


def test_when_predict_with_linear_regressor_with_given_parameters_then_returns_expected_results():
    mdl = create_linear_regressor_with_given_parameters([2], 4)

    assert mdl.predict(np.array([0])) == approx(4)
    assert mdl.predict(np.array([1])) == approx(6)
    assert mdl.predict(np.array([0, 1, 2, 3])) == approx([4, 6, 8, 10])
