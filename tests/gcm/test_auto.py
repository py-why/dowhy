import networkx as nx
import numpy as np
import pandas as pd
from _pytest.python_api import approx
from flaky import flaky
from pytest import mark
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from dowhy.gcm import ProbabilisticCausalModel, draw_samples, fit
from dowhy.gcm.auto import AssignmentQuality, assign_causal_mechanisms, has_linear_relationship


def _generate_linear_regression_data(num_samples=1000):
    X = np.random.normal(0, 1, (num_samples, 5))
    Y = np.sum(X * np.random.uniform(-5, 5, X.shape[1]), axis=1)

    return X, Y


def _generate_non_linear_regression_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = np.sum(np.log(abs(X)), axis=1)

    return X, Y


def _generate_linear_classification_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = (np.sum(X * np.random.uniform(-5, 5, X.shape[1]), axis=1) > 0).astype(str)

    return X, Y


def _generate_non_classification_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = (np.sum(np.exp(X), axis=1) > np.median(np.sum(np.exp(X), axis=1))).astype(str)

    return X, Y


@flaky(max_runs=3)
def test_given_linear_regression_problem_when_auto_assign_causal_models_with_good_quality_returns_linear_model():
    X, Y = _generate_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(
        causal_model.causal_mechanism("Y").prediction_model.sklearn_model, LinearRegression
    ) or isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, Pipeline)


@flaky(max_runs=3)
def test_given_linear_regression_problem_when_auto_assign_causal_models_with_better_quality_returns_linear_model():
    X, Y = _generate_linear_regression_data(5000)

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert isinstance(
        causal_model.causal_mechanism("Y").prediction_model.sklearn_model, LinearRegression
    ) or isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, Pipeline)


@flaky(max_runs=3)
def test_given_non_linear_regression_problem_when_auto_assign_causal_models_with_good_quality_returns_non_linear_model():
    X, Y = _generate_non_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(
        causal_model.causal_mechanism("Y").prediction_model.sklearn_model, HistGradientBoostingRegressor
    ) or isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, Pipeline)


@flaky(max_runs=3)
def test_given_non_linear_regression_problem_when_auto_assign_causal_models_with_better_quality_returns_non_linear_model():
    X, Y = _generate_non_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert not isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, LinearRegression)
    assert not isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, LassoCV)
    assert not isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, ElasticNetCV)
    assert not isinstance(causal_model.causal_mechanism("Y").prediction_model.sklearn_model, RidgeCV)


@flaky(max_runs=3)
def test_given_linear_classification_problem_when_auto_assign_causal_models_with_good_quality_returns_linear_model():
    X, Y = _generate_linear_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism("Y").classifier_model.sklearn_model, LogisticRegression)


@flaky(max_runs=3)
def test_given_linear_classification_problem_when_auto_assign_causal_models_with_better_quality_returns_linear_model():
    X, Y = _generate_linear_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert isinstance(causal_model.causal_mechanism("Y").classifier_model.sklearn_model, LogisticRegression)


@flaky(max_runs=3)
def test_given_non_linear_classification_problem_when_auto_assign_causal_models_with_good_quality_returns_non_linear_model():
    X, Y = _generate_non_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(
        causal_model.causal_mechanism("Y").classifier_model.sklearn_model, HistGradientBoostingClassifier
    ) or isinstance(causal_model.causal_mechanism("Y").classifier_model.sklearn_model, Pipeline)


@flaky(max_runs=3)
def test_given_non_linear_classification_problem_when_auto_assign_causal_models_with_better_quality_returns_non_linear_model():
    X, Y = _generate_non_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert not isinstance(causal_model.causal_mechanism("Y").classifier_model.sklearn_model, LogisticRegression)
    assert not isinstance(causal_model.causal_mechanism("Y").classifier_model.sklearn_model, GaussianNB)


@flaky(max_runs=3)
def test_given_polynomial_regression_data_with_categorical_input_when_auto_assign_causal_models_then_does_not_raise_error():
    X = np.column_stack(
        [np.random.choice(2, 100, replace=True).astype(str), np.random.normal(0, 1, (100, 2)).astype(object)]
    ).astype(object)
    Y = []
    for i in range(X.shape[0]):
        Y.append(X[i, 1] * X[i, 2] if X[i, 0] == "0" else X[i, 1] + X[i, 2])

    Y = np.array(Y)

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y")]))
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER, override_models=True)


@flaky(max_runs=3)
def test_given_polynomial_classification_data_with_categorical_input_when_auto_assign_causal_models_then_does_not_raise_error():
    X = np.random.normal(0, 1, (100, 2))
    Y = []

    for x in X:
        if x[0] * x[1] > 0:
            Y.append("Class 0")
        else:
            Y.append("Class 1")

    Y = np.array(Y)

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "Y"), ("X1", "Y")]))
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD, override_models=True)


def test_when_auto_called_from_main_namespace_returns_no_attribute_error():
    from dowhy import gcm

    _ = gcm.auto.AssignmentQuality.GOOD


@mark.skip("Not running AutoGluon-based tests as part of CI yet.")
def test_when_using_best_quality_then_returns_auto_gluon_model():
    from dowhy.gcm.ml import AutoGluonClassifier, AutoGluonRegressor

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X", "Y")]))

    assign_causal_mechanisms(causal_model, pd.DataFrame({"X": [1], "Y": [1]}), quality=AssignmentQuality.BEST)
    assert isinstance(causal_model.causal_mechanism("Y").prediction_model, AutoGluonRegressor)

    assign_causal_mechanisms(
        causal_model, pd.DataFrame({"X": [1], "Y": ["Class 1"]}), quality=AssignmentQuality.BEST, override_models=True
    )
    assert isinstance(causal_model.causal_mechanism("Y").classifier_model, AutoGluonClassifier)


@flaky(max_runs=3)
def test_given_linear_gaussian_data_when_fit_scm_with_auto_assigned_models_with_default_parameters_then_generate_samples_with_correct_statistics():
    X0 = np.random.normal(0, 1, 2000)
    X1 = 2 * X0 + np.random.normal(0, 0.2, 2000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.2, 2000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.2, 2000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))

    assign_causal_mechanisms(causal_model, original_observations)

    fit(causal_model, original_observations)
    generated_samples = draw_samples(causal_model, 2000)

    assert np.mean(generated_samples["X0"]) == approx(np.mean(X0), abs=0.1)
    assert np.std(generated_samples["X0"]) == approx(np.std(X0), abs=0.1)
    assert np.mean(generated_samples["X1"]) == approx(np.mean(X1), abs=0.1)
    assert np.std(generated_samples["X1"]) == approx(np.std(X1), abs=0.1)
    assert np.mean(generated_samples["X2"]) == approx(np.mean(X2), abs=0.1)
    assert np.std(generated_samples["X2"]) == approx(np.std(X2), abs=0.1)
    assert np.mean(generated_samples["X3"]) == approx(np.mean(X3), abs=0.1)
    assert np.std(generated_samples["X3"]) == approx(np.std(X3), abs=0.1)


@flaky(max_runs=3)
def test_given_nonlinear_gaussian_data_when_fit_scm_with_auto_assigned_models_with_default_parameters_then_generate_samples_with_correct_statistics():
    X0 = np.random.normal(0, 1, 2000)
    X1 = np.sin(2 * X0) + np.random.normal(0, 0.2, 2000)
    X2 = 0.5 * X0**2 + np.random.normal(0, 0.2, 2000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.2, 2000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))

    assign_causal_mechanisms(causal_model, original_observations)

    fit(causal_model, original_observations)
    generated_samples = draw_samples(causal_model, 2000)

    assert np.mean(generated_samples["X0"]) == approx(np.mean(X0), abs=0.1)
    assert np.std(generated_samples["X0"]) == approx(np.std(X0), abs=0.1)
    assert np.mean(generated_samples["X1"]) == approx(np.mean(X1), abs=0.1)
    assert np.std(generated_samples["X1"]) == approx(np.std(X1), abs=0.1)
    assert np.mean(generated_samples["X2"]) == approx(np.mean(X2), abs=0.1)
    assert np.std(generated_samples["X2"]) == approx(np.std(X2), abs=0.1)
    assert np.mean(generated_samples["X3"]) == approx(np.mean(X3), abs=0.1)
    assert np.std(generated_samples["X3"]) == approx(np.std(X3), abs=0.1)


def test_givne_simple_data_when_apply_has_linear_relationship_then_returns_expected_results():
    X = np.random.random(1000)

    assert has_linear_relationship(X, 2 * X)
    assert not has_linear_relationship(X, X**2)


@flaky(max_runs=3)
def test_given_categorical_data_when_calling_has_linear_relationship_then_returns_correct_results():
    X1 = np.random.normal(0, 1, 1000)
    X2 = np.random.normal(0, 1, 1000)

    assert has_linear_relationship(np.column_stack([X1, X2]), (X1 + X2 > 0).astype(str))
    assert not has_linear_relationship(np.column_stack([X1, X2]), (X1 * X2 > 0).astype(str))


def test_given_imbalanced_categorical_data_when_calling_has_linear_relationship_then_does_not_raise_exception():
    X = np.random.normal(0, 1, 1000)
    Y = np.array(["OneClass"] * 1000)

    assert has_linear_relationship(np.append(X, 0), np.append(Y, "RareClass"))

    X = np.random.normal(0, 1, 100000)
    Y = np.array(["OneClass"] * 100000)

    assert has_linear_relationship(
        np.append(X, np.random.normal(0, 0.000001, 100)), np.append(Y, np.array(["RareClass"] * 100))
    )


def test_given_data_with_rare_categorical_features_when_calling_has_linear_relationship_then_does_not_raise_exception():
    X = np.array(["Feature" + str(i) for i in range(20)])
    Y = np.append(np.array(["Class1"] * 10), np.array(["Class2"] * 10))

    assert has_linear_relationship(X, Y)
