import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx
from sklearn.linear_model import LogisticRegression

from dowhy.gcm import StructuralCausalModel, auto, fit, intrinsic_causal_influence
from dowhy.gcm._noise import noise_samples_of_ancestors
from dowhy.gcm.graph import node_connected_subgraph_view
from dowhy.gcm.ml import create_hist_gradient_boost_classifier
from dowhy.gcm.uncertainty import estimate_entropy_of_probabilities, estimate_variance
from dowhy.gcm.util.general import apply_one_hot_encoding, fit_one_hot_encoders


@flaky(max_runs=3)
def test_intrinsic_causal_influence_variance_linear():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + np.random.normal(0, 0.001, 10000)
    X2 = X1 + np.random.normal(0, 2, 10000)
    X3 = X2 + np.random.normal(0, 1, 10000)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx")
    assert iccs["X0"] == approx(1, abs=0.25)
    assert iccs["X1"] == approx(0, abs=0.05)
    assert iccs["X2"] == approx(4, abs=0.5)
    assert iccs["X3"] == approx(1, abs=0.25)
    assert np.sum([iccs[key] for key in iccs]) == approx(estimate_variance(X3), abs=0.5)

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact")
    assert iccs["X0"] == approx(1, abs=0.25)
    assert iccs["X1"] == approx(0, abs=0.05)
    assert iccs["X2"] == approx(4, abs=0.5)
    assert iccs["X3"] == approx(1, abs=0.25)
    assert np.sum([iccs[key] for key in iccs]) == approx(estimate_variance(X3), abs=0.5)


@flaky(max_runs=3)
def test_intrinsic_causal_influence_categorical():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    N1 = np.random.normal(0, 0.001, 10000)
    N2 = np.random.normal(0, 2, 10000)
    N3 = np.random.normal(0, 1, 10000)
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + N1
    X2 = X1 + N2
    X3 = ((X2 + N3) >= 0).astype(str)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    log_mdl = LogisticRegression(max_iter=1000)
    log_mdl.fit(np.column_stack([X0, N1, N2, N3]), X3)

    expected_output_full_subset = -estimate_entropy_of_probabilities(
        log_mdl.predict_proba(np.column_stack([X0, N1, N2, N3]))
    )
    expected_output_empty_subset = -estimate_entropy_of_probabilities(np.array([[0.5, 0.5]]))

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.14, abs=0.05)
    assert iccs["X1"] == approx(0.01, abs=0.05)
    assert iccs["X2"] == approx(0.38, abs=0.05)
    assert iccs["X3"] == approx(0.12, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(
        expected_output_full_subset - expected_output_empty_subset, abs=0.05
    )

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.14, abs=0.05)
    assert iccs["X1"] == approx(0.01, abs=0.05)
    assert iccs["X2"] == approx(0.38, abs=0.05)
    assert iccs["X3"] == approx(0.12, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(-expected_output_empty_subset, abs=0.05)


@flaky(max_runs=3)
def test_intrinsic_causal_influence_categorical_2():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    N1 = np.random.normal(0, 0.001, 10000)
    N2 = np.random.normal(0, 2, 10000)
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + N1
    X2 = X1 + N2

    X3 = []
    for i in range(10000):
        rand_val = np.random.choice(3, 1).squeeze()
        if rand_val == 0:
            X3.append(X2[i] + np.random.normal(-5, 0.5, 1))
        elif rand_val == 1:
            X3.append(X2[i] + np.random.normal(10, 0.5, 1))
        else:
            X3.append(X2[i] + np.random.normal(5, 0.5, 1))

    X3 = np.array(X3)
    X3 = (X3 > np.median(X3)).astype(str)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3.reshape(-1)})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    data_samples, noise_samples = noise_samples_of_ancestors(
        StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, "X3")), "X3", 100000
    )
    X = apply_one_hot_encoding(noise_samples.to_numpy(), fit_one_hot_encoders(noise_samples.to_numpy()))
    log_mdl = LogisticRegression()
    log_mdl.fit(X, data_samples["X3"].to_numpy())

    expected_output_full_subset = -estimate_entropy_of_probabilities(log_mdl.predict_proba(X))
    expected_output_empty_subset = -estimate_entropy_of_probabilities(np.array([[0.5, 0.5]]))

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.05, abs=0.05)
    assert iccs["X1"] == approx(0.03, abs=0.05)
    assert iccs["X2"] == approx(0.1, abs=0.05)
    assert iccs["X3"] == approx(0.5, abs=0.1)

    assert np.sum([iccs[key] for key in iccs]) == approx(
        expected_output_full_subset - expected_output_empty_subset, abs=0.05
    )

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.05, abs=0.05)
    assert iccs["X1"] == approx(0.03, abs=0.05)
    assert iccs["X2"] == approx(0.1, abs=0.05)
    assert iccs["X3"] == approx(0.5, abs=0.1)

    assert np.sum([iccs[key] for key in iccs]) == approx(-expected_output_empty_subset, abs=0.05)


@flaky(max_runs=3)
def test_given_only_categorical_data_when_estimate_icc_then_does_not_fail():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + np.random.normal(0, 0.001, 10000)
    X2 = X1 + np.random.normal(0, 2, 10000)
    X3 = ((X2 + np.random.normal(0, 1, 10000)) >= 0).astype(str)
    training_data = pd.DataFrame(
        {"X0": (X0 >= 0).astype(str), "X1": (X1 >= 0).astype(str), "X2": (X2 >= 0).astype(str), "X3": X3}
    )
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    data_samples, noise_samples = noise_samples_of_ancestors(
        StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, "X3")), "X3", 100000
    )
    X = apply_one_hot_encoding(noise_samples.to_numpy(), fit_one_hot_encoders(noise_samples.to_numpy()))
    log_mdl = create_hist_gradient_boost_classifier()  # Due to the categorical features, this becomes non-linear.
    log_mdl.fit(X, data_samples["X3"].to_numpy())

    expected_output_full_subset = -estimate_entropy_of_probabilities(log_mdl.predict_probabilities(X))
    expected_output_empty_subset = -estimate_entropy_of_probabilities(np.array([[0.5, 0.5]]))

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.08, abs=0.03)
    assert iccs["X1"] == approx(0, abs=0.005)
    assert iccs["X2"] == approx(0.33, abs=0.05)
    assert iccs["X3"] == approx(0.27, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(
        expected_output_full_subset - expected_output_empty_subset, abs=0.05
    )

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.08, abs=0.03)
    assert iccs["X1"] == approx(0, abs=0.005)
    assert iccs["X2"] == approx(0.33, abs=0.05)
    assert iccs["X3"] == approx(0.27, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(-expected_output_empty_subset, abs=0.05)


def test_when_calling_intrinsic_causal_influence_then_the_shape_of_inputs_in_the_attribution_function_should_be_equal():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1")]))
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + np.random.normal(0, 0.001, 10000)
    training_data = pd.DataFrame({"X0": X0, "X1": X1})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    def my_attr_func(X, Y):
        assert np.all(X.shape == Y.shape)
        return 0

    intrinsic_causal_influence(causal_model, "X1", attribution_func=my_attr_func)
