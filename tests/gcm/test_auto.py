import networkx as nx
import numpy as np
import pandas as pd
import pytest
from _pytest.python_api import approx
from flaky import flaky
from pytest import mark
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from dowhy import gcm
from dowhy.gcm import (
    AdditiveNoiseModel,
    DiscreteAdditiveNoiseModel,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    StructuralCausalModel,
    counterfactual_samples,
    draw_samples,
    fit,
)
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


def test_given_continuous_and_discrete_data_when_auto_assign_then_correct_assigns_discrete_anm():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X", "Y"), ("Y", "Z")]))
    data = {
        "X": np.random.normal(0, 1, 100),
        "Y": np.random.choice(2, 100, replace=True),
        "Z": np.random.normal(0, 1, 100),
    }

    assign_causal_mechanisms(causal_model, pd.DataFrame(data), quality=AssignmentQuality.GOOD)
    assert isinstance(causal_model.causal_mechanism("X"), EmpiricalDistribution)
    assert isinstance(causal_model.causal_mechanism("Y"), DiscreteAdditiveNoiseModel)
    assert isinstance(causal_model.causal_mechanism("Z"), AdditiveNoiseModel)


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


@flaky(max_runs=2)
def test_given_continuous_data_when_print_auto_summary_then_returns_expected_formats():
    X, Y = _generate_non_linear_regression_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    summary_result = assign_causal_mechanisms(causal_model, pd.DataFrame(data))
    summary_string = str(summary_result)

    assert "X0" in summary_result._nodes
    assert "X1" in summary_result._nodes
    assert "X2" in summary_result._nodes
    assert "X3" in summary_result._nodes
    assert "X4" in summary_result._nodes
    assert "Y" in summary_result._nodes

    assert len(summary_result._nodes["X0"]["model_performances"]) == 0
    assert len(summary_result._nodes["X1"]["model_performances"]) == 0
    assert len(summary_result._nodes["X2"]["model_performances"]) == 0
    assert len(summary_result._nodes["X3"]["model_performances"]) == 0
    assert len(summary_result._nodes["X4"]["model_performances"]) == 0
    assert len(summary_result._nodes["Y"]["model_performances"]) > 0

    assert (
        """When using this auto assignment function, the given data is used to automatically assign a causal mechanism to each node. Note that causal mechanisms can also be customized and assigned manually.
The following types of causal mechanisms are considered for the automatic selection:

If root node:
An empirical distribution, i.e., the distribution is represented by randomly sampling from the provided data. This provides a flexible and non-parametric way to model the marginal distribution and is valid for all types of data modalities.

If non-root node and the data is continuous:
Additive Noise Models (ANM) of the form X_i = f(PA_i) + N_i, where PA_i are the parents of X_i and the unobserved noise N_i is assumed to be independent of PA_i.To select the best model for f, different regression models are evaluated and the model with the smallest mean squared error is selected.Note that minimizing the mean squared error here is equivalent to selecting the best choice of an ANM.

If non-root node and the data is discrete:
Discrete Additive Noise Models have almost the same definition as non-discrete ANMs, but come with an additional constraint for f to only return discrete values.
Note that 'discrete' here refers to numerical values with an order. If the data is categorical, consider representing them as strings to ensure proper model selection.

If non-root node and the data is categorical:
A functional causal model based on a classifier, i.e., X_i = f(PA_i, N_i).
Here, N_i follows a uniform distribution on [0, 1] and is used to randomly sample a class (category) using the conditional probability distribution produced by a classification model.Here, different model classes are evaluated using the (negative) F1 score and the best performing model class is selected.

In total, 6 nodes were analyzed:

--- Node: X0
Node X0 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X1
Node X1 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X2
Node X2 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X3
Node X3 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X4
Node X4 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: Y
Node Y is a non-root node with continuous data. Assigning 'AdditiveNoiseModel using """
        in summary_string
    )
    assert "This represents the causal relationship as Y := f(X0,X1,X2,X3,X4) + N." in summary_string
    assert "For the model selection, the following models were evaluated on the mean squared error (MSE) metric:"
    assert (
        """===Note===
Note, based on the selected auto assignment quality, the set of evaluated models changes.
For more insights toward the quality of the fitted graphical causal model, consider using the evaluate_causal_model function after fitting the causal mechanisms."""
        in summary_string
    )


@flaky(max_runs=2)
def test_given_categorical_data_when_print_auto_summary_then_returns_expected_formats():
    X, Y = _generate_linear_classification_data()

    causal_model = ProbabilisticCausalModel(
        nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")])
    )
    data = {"X" + str(i): X[:, i] for i in range(X.shape[1])}
    data.update({"Y": Y})

    summary_result = assign_causal_mechanisms(causal_model, pd.DataFrame(data))
    summary_string = str(summary_result)

    assert "X0" in summary_result._nodes
    assert "X1" in summary_result._nodes
    assert "X2" in summary_result._nodes
    assert "X3" in summary_result._nodes
    assert "X4" in summary_result._nodes
    assert "Y" in summary_result._nodes

    assert len(summary_result._nodes["X0"]["model_performances"]) == 0
    assert len(summary_result._nodes["X1"]["model_performances"]) == 0
    assert len(summary_result._nodes["X2"]["model_performances"]) == 0
    assert len(summary_result._nodes["X3"]["model_performances"]) == 0
    assert len(summary_result._nodes["X4"]["model_performances"]) == 0
    assert len(summary_result._nodes["Y"]["model_performances"]) > 0

    assert (
        """The following types of causal mechanisms are considered for the automatic selection:

If root node:
An empirical distribution, i.e., the distribution is represented by randomly sampling from the provided data. This provides a flexible and non-parametric way to model the marginal distribution and is valid for all types of data modalities.

If non-root node and the data is continuous:
Additive Noise Models (ANM) of the form X_i = f(PA_i) + N_i, where PA_i are the parents of X_i and the unobserved noise N_i is assumed to be independent of PA_i.To select the best model for f, different regression models are evaluated and the model with the smallest mean squared error is selected.Note that minimizing the mean squared error here is equivalent to selecting the best choice of an ANM.

If non-root node and the data is discrete:
Discrete Additive Noise Models have almost the same definition as non-discrete ANMs, but come with an additional constraint for f to only return discrete values.
Note that 'discrete' here refers to numerical values with an order. If the data is categorical, consider representing them as strings to ensure proper model selection.

If non-root node and the data is categorical:
A functional causal model based on a classifier, i.e., X_i = f(PA_i, N_i).
Here, N_i follows a uniform distribution on [0, 1] and is used to randomly sample a class (category) using the conditional probability distribution produced by a classification model.Here, different model classes are evaluated using the (negative) F1 score and the best performing model class is selected.

In total, 6 nodes were analyzed:

--- Node: X0
Node X0 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X1
Node X1 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X2
Node X2 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X3
Node X3 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: X4
Node X4 is a root node. Therefore, assigning 'Empirical Distribution' to the node representing the marginal distribution.

--- Node: Y
Node Y is a non-root node with categorical data. Assigning 'Classifier FCM based on """
        in summary_string
    )
    assert "This represents the causal relationship as Y := f(X0,X1,X2,X3,X4,N)." in summary_string
    assert "For the model selection, the following models were evaluated on the (negative) F1 metric:" in summary_string
    assert (
        """===Note===
Note, based on the selected auto assignment quality, the set of evaluated models changes.
For more insights toward the quality of the fitted graphical causal model, consider using the evaluate_causal_model function after fitting the causal mechanisms."""
        in summary_string
    )


def test_given_imbalanced_classes_when_auto_assign_mechanism_then_handles_as_expected():
    X = np.random.normal(0, 1, 1000)
    Y = np.array(["OneClass"] * 1000)

    with pytest.raises(ValueError):
        assign_causal_mechanisms(StructuralCausalModel(nx.DiGraph([("X", "Y")])), pd.DataFrame({"X": X, "Y": Y}))

    # Having at least one sample from the second class should not raise an error.
    X = np.append(X, 0)
    Y = np.append(Y, "RareClass")

    assign_causal_mechanisms(StructuralCausalModel(nx.DiGraph([("X", "Y")])), pd.DataFrame({"X": X, "Y": Y}))


@flaky(max_runs=2)
def test_given_missing_data_only_numerical_when_auto_assign_mechanism_with_experimental_feature_then_handles_as_expected():
    X = np.random.normal(0, 5, 5000)
    Y = 2 * X + 10 + np.random.normal(0, 0.05, 5000)
    Z = X + Y + 20 + np.random.normal(0, 0.05, 5000)

    data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    mask = np.random.random(data.shape) >= 0.25  # 25% missing data
    mask[:, 1] = 1  # Ensure that the categorical feature is not missing as this is not supported yet.

    data_with_nas = data.mask(~mask)
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X", "Y"), ("Y", "Z"), ("X", "Z")]))

    with pytest.raises(ValueError):
        # Raise error when experimental flag is not turned on
        assign_causal_mechanisms(causal_model, data_with_nas)

    assign_causal_mechanisms(causal_model, data_with_nas, experimental_allow_nans=True)
    fit(causal_model, data_with_nas)

    drawn_samples = gcm.draw_samples(causal_model, 5000)

    assert drawn_samples["X"].mean() == approx(0, abs=1)
    assert drawn_samples["Y"].mean() == approx(10, abs=2)
    assert drawn_samples["Z"].mean() == approx(30, abs=3)

    interventional_drawn_samples = gcm.interventional_samples(
        causal_model, {"X": lambda x: 10}, num_samples_to_draw=100
    )
    assert interventional_drawn_samples["X"].to_numpy() == approx(np.array([10] * 100))
    assert interventional_drawn_samples["Y"].mean() == approx(30, abs=5)
    assert interventional_drawn_samples["Z"].mean() == approx(60, abs=5)

    counterfactual_computed_samples = counterfactual_samples(
        causal_model, {"X": lambda x: 10}, noise_data=pd.DataFrame({"X": [10], "Y": [1], "Z": [10]})
    )
    assert counterfactual_computed_samples["X"][0] == 10
    assert counterfactual_computed_samples["Y"][0] == approx(30, abs=5)
    assert counterfactual_computed_samples["Z"][0] == approx(70, abs=5)

    # Just check if it doesn't raise errors.
    gcm.intrinsic_causal_influence(causal_model, "Z")
    gcm.attribute_anomalies(causal_model, "Y", data_with_nas.iloc[:5])


@flaky(max_runs=2)
def test_given_missing_data_mixed_numerical_and_categorical_when_auto_assign_mechanism_with_experimental_feature_then_handles_as_expected():
    X = np.random.normal(0, 5, 5000)
    Y = []
    Z = []

    for x in X:
        if x < 0:
            Y.append("Class 0")
        else:
            Y.append("Class 1")

    for i in range(X.shape[0]):
        Z.append(X[i] + (-10 if Y[i] == "Class 0" else 10) + np.random.normal(0, 0.1))

    data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    mask = np.random.random(data.shape) >= 0.25  # 25% missing data
    mask[:, 1] = 1  # Ensure that the categorical feature is not missing as this is not supported yet.

    data_with_nas = data.mask(~mask)
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X", "Y"), ("Y", "Z"), ("X", "Z")]))

    with pytest.raises(ValueError):
        # Raise error when experimental flag is not turned on
        assign_causal_mechanisms(causal_model, data_with_nas)

    assign_causal_mechanisms(causal_model, data_with_nas, experimental_allow_nans=True)
    fit(causal_model, data_with_nas)

    drawn_samples = gcm.draw_samples(causal_model, 5000)

    assert drawn_samples["X"].mean() == approx(0, abs=1)
    assert np.sum(drawn_samples["Y"] == "Class 0") == approx(X.shape[0] // 2, abs=500)
    assert drawn_samples["Z"].mean() == approx(0, abs=1)

    interventional_drawn_samples = gcm.interventional_samples(causal_model, {"X": lambda x: 10}, num_samples_to_draw=10)
    assert interventional_drawn_samples["X"].to_numpy() == approx(np.array([10] * 10))
    assert np.sum(interventional_drawn_samples["Y"] == "Class 1") == approx(10, abs=3)
    assert np.mean(interventional_drawn_samples["Z"].to_numpy()) == approx(20, abs=5)

    counterfactual_computed_samples = counterfactual_samples(
        causal_model, {"X": lambda x: 10}, noise_data=pd.DataFrame({"X": [10], "Y": [1], "Z": [10]})
    )
    assert counterfactual_computed_samples["X"][0] == 10
    assert counterfactual_computed_samples["Y"][0] == "Class 1"
    assert counterfactual_computed_samples["Z"][0] == approx(30, abs=5)

    # Just check if it doesn't raise errors.
    gcm.intrinsic_causal_influence(causal_model, "Z")
