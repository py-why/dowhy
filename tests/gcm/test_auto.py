import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB

from dowhy.gcm import ProbabilisticCausalModel
from dowhy.gcm.auto import AssignmentQuality, assign_causal_mechanisms


def __generate_linear_regression_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = np.sum(X * np.random.uniform(-5, 5, X.shape[1]), axis=1)

    return X, Y


def __generate_non_linear_regression_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = np.sum(X ** 2, axis=1)

    return X, Y


def __generate_linear_classification_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = (np.sum(X * np.random.uniform(-5, 5, X.shape[1]), axis=1) > 0).astype(str)

    return X, Y


def __generate_non_classification_data():
    X = np.random.normal(0, 1, (1000, 5))
    Y = (np.sum(np.exp(X), axis=1) > np.median(np.sum(np.exp(X), axis=1))).astype(str)

    return X, Y


@flaky(max_runs=3)
def test_given_linear_regression_problem_when_auto_assign_causal_models_with_good_quality_returns_linear_model():
    X, Y = __generate_linear_regression_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4',
                                                                                                             'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, LinearRegression)


@flaky(max_runs=3)
def test_given_linear_regression_problem_when_auto_assign_causal_models_with_better_quality_returns_linear_model():
    X, Y = __generate_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, LinearRegression)


@flaky(max_runs=3)
def test_given_non_linear_regression_problem_when_auto_assign_causal_models_with_good_quality_returns_non_linear_model():
    X, Y = __generate_non_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, HistGradientBoostingRegressor)


@flaky(max_runs=3)
def test_given_non_linear_regression_problem_when_auto_assign_causal_models_with_better_quality_returns_non_linear_model():
    X, Y = __generate_non_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert not isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, LinearRegression)
    assert not isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, LassoCV)
    assert not isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, ElasticNetCV)
    assert not isinstance(causal_model.causal_mechanism('Y').prediction_model.sklearn_model, RidgeCV)


@flaky(max_runs=3)
def test_given_linear_classification_problem_when_auto_assign_causal_models_with_good_quality_returns_linear_model():
    X, Y = __generate_linear_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism('Y').classifier_model.sklearn_model, LogisticRegression)


@flaky(max_runs=3)
def test_given_linear_classification_problem_when_auto_assign_causal_models_with_better_quality_returns_linear_model():
    X, Y = __generate_linear_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert isinstance(causal_model.causal_mechanism('Y').classifier_model.sklearn_model, LogisticRegression)


@flaky(max_runs=3)
def test_given_non_linear_classification_problem_when_auto_assign_causal_models_with_good_quality_returns_non_linear_model():
    X, Y = __generate_non_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism('Y').classifier_model.sklearn_model, HistGradientBoostingClassifier)


@flaky(max_runs=3)
def test_given_non_linear_classification_problem_when_auto_assign_causal_models_with_better_quality_returns_non_linear_model():
    X, Y = __generate_non_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([('X0', 'Y'), ('X1', 'Y'), ('X2', 'Y'), ('X3', 'Y'), ('X4', 'Y')]))
    data = {'X' + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({'Y': Y})

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.BETTER)
    assert not isinstance(causal_model.causal_mechanism('Y').classifier_model.sklearn_model, LogisticRegression)
    assert not isinstance(causal_model.causal_mechanism('Y').classifier_model.sklearn_model, GaussianNB)


def test_when_auto_called_from_main_namespace_returns_no_attribute_error():
    from dowhy import gcm
    _ = gcm.auto.AssignmentQuality.GOOD
