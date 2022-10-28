import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky

from dowhy.gcm import (
    InvertibleStructuralCausalModel,
    RejectionResult,
    auto,
    fit,
    kernel_based,
    refute_causal_structure,
    refute_invertible_model,
)


def _generate_simple_non_linear_data() -> pd.DataFrame:
    X = np.random.normal(loc=0, scale=1, size=5000)
    Y = X**2 + np.random.normal(loc=0, scale=1, size=5000)
    Z = np.exp(-Y) + np.random.normal(loc=0, scale=1, size=5000)

    return pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))


@flaky(max_runs=5)
def test_refute_causal_structure_collider():
    # collider: X->Z<-Y
    collider_dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
    X = np.random.normal(size=500)
    Y = np.random.normal(size=500)
    Z = 2 * X + 3 * Y + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    rejection_result, rejection_summary = refute_causal_structure(collider_dag, data)

    assert rejection_result == RejectionResult.NOT_REJECTED
    assert rejection_summary["X"]["local_markov_test"] == dict()
    assert rejection_summary["X"]["edge_dependence_test"] == dict()
    assert rejection_summary["Y"]["local_markov_test"] == dict()
    assert rejection_summary["Y"]["edge_dependence_test"] == dict()
    assert rejection_summary["Z"]["local_markov_test"] == dict()
    assert rejection_summary["Z"]["edge_dependence_test"]["X"]["success"] == True
    assert rejection_summary["Z"]["edge_dependence_test"]["Y"]["success"] == True


@flaky(max_runs=5)
def test_refute_causal_structure_chain():
    # chain: X->Z->Y
    chain_dag = nx.DiGraph([("X", "Z"), ("Z", "Y")])
    X = np.random.normal(size=500)
    Z = 2 * X + np.random.normal(size=500)
    Y = 3 * Z + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    rejection_result, rejection_summary = refute_causal_structure(chain_dag, data)

    assert rejection_result == RejectionResult.NOT_REJECTED
    assert rejection_summary["X"]["local_markov_test"] == dict()
    assert rejection_summary["X"]["edge_dependence_test"] == dict()
    assert rejection_summary["Z"]["local_markov_test"] == dict()
    assert rejection_summary["Z"]["edge_dependence_test"]["X"]["success"] == True
    assert rejection_summary["Y"]["local_markov_test"]["success"] == True
    assert rejection_summary["Y"]["edge_dependence_test"]["Z"]["success"] == True


@flaky(max_runs=5)
def test_refute_causal_structure_fork():
    # fork: X<-Z->Y
    fork_dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
    Z = np.random.normal(size=500)
    X = 2 * Z + np.random.normal(size=500)
    Y = 3 * Z + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    rejection_result, rejection_summary = refute_causal_structure(fork_dag, data)

    assert rejection_result == RejectionResult.NOT_REJECTED
    assert rejection_summary["Z"]["local_markov_test"] == dict()
    assert rejection_summary["Z"]["edge_dependence_test"] == dict()
    assert rejection_summary["X"]["local_markov_test"]["success"] == True
    assert rejection_summary["X"]["edge_dependence_test"]["Z"]["success"] == True
    assert rejection_summary["Y"]["local_markov_test"]["success"] == True
    assert rejection_summary["Y"]["edge_dependence_test"]["Z"]["success"] == True


@flaky(max_runs=5)
def test_refute_causal_structure_general():
    # general DAG: X<-Z->Y, X->Y
    general_dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
    Z = np.random.normal(size=500)
    X = 2 * Z + np.random.normal(size=500)
    Y = 2 * Z + 3 * X + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    rejection_result, rejection_summary = refute_causal_structure(general_dag, data)

    assert rejection_result == RejectionResult.NOT_REJECTED
    assert rejection_summary["Z"]["local_markov_test"] == dict()
    assert rejection_summary["Z"]["edge_dependence_test"] == dict()
    assert rejection_summary["X"]["local_markov_test"] == dict()
    assert rejection_summary["X"]["edge_dependence_test"]["Z"]["success"] == True
    assert rejection_summary["Y"]["local_markov_test"] == dict()
    assert rejection_summary["Y"]["edge_dependence_test"]["Z"]["success"] == True
    assert rejection_summary["Y"]["edge_dependence_test"]["X"]["success"] == True


@flaky(max_runs=5)
def test_refute_causal_structure_adjusted_p_values():
    # fork: X<-Z->Y
    fork_dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
    Z = np.random.normal(size=500)
    X = 2 * Z + np.random.normal(size=500)
    Y = 3 * Z + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    rejection_result, rejection_summary = refute_causal_structure(fork_dag, data, fdr_control_method="fdr_bh")

    assert (
        rejection_summary["X"]["local_markov_test"]["fdr_adjusted_p_value"]
        >= rejection_summary["X"]["local_markov_test"]["p_value"]
    )
    assert (
        rejection_summary["X"]["edge_dependence_test"]["Z"]["fdr_adjusted_p_value"]
        >= rejection_summary["X"]["edge_dependence_test"]["Z"]["p_value"]
    )
    assert (
        rejection_summary["Y"]["local_markov_test"]["fdr_adjusted_p_value"]
        >= rejection_summary["Y"]["local_markov_test"]["p_value"]
    )
    assert (
        rejection_summary["Y"]["edge_dependence_test"]["Z"]["fdr_adjusted_p_value"]
        >= rejection_summary["Y"]["edge_dependence_test"]["Z"]["p_value"]
    )


def test_when_using_refute_causal_structure_without_fdrc_then_nans_for_adjusted_p_values_are_returned():
    fork_dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
    Z = np.random.normal(size=500)
    X = 2 * Z + np.random.normal(size=500)
    Y = 3 * Z + np.random.normal(size=500)
    data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
    _, rejection_summary = refute_causal_structure(fork_dag, data, fdr_control_method=None)

    assert np.isnan(rejection_summary["X"]["local_markov_test"]["fdr_adjusted_p_value"])
    assert np.isnan(rejection_summary["X"]["edge_dependence_test"]["Z"]["fdr_adjusted_p_value"])
    assert np.isnan(rejection_summary["Y"]["local_markov_test"]["fdr_adjusted_p_value"])
    assert np.isnan(rejection_summary["Y"]["edge_dependence_test"]["Z"]["fdr_adjusted_p_value"])

    assert not np.isnan(rejection_summary["X"]["local_markov_test"]["p_value"])
    assert not np.isnan(rejection_summary["X"]["edge_dependence_test"]["Z"]["p_value"])
    assert not np.isnan(rejection_summary["Y"]["local_markov_test"]["p_value"])
    assert not np.isnan(rejection_summary["Y"]["edge_dependence_test"]["Z"]["p_value"])


@flaky(max_runs=2)
def test_given_non_linear_data_and_correct_dag_when_refute_invertible_model_then_not_reject_model():
    data = _generate_simple_non_linear_data()

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X", "Y"), ("Y", "Z")]))  # X->Y->Z
    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)

    fit(causal_model, data)

    assert (
        refute_invertible_model(
            causal_model, data, independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=5)
        )
        == RejectionResult.NOT_REJECTED
    )
    assert (
        refute_invertible_model(
            causal_model,
            data,
            independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=5),
            fdr_control_method="fdr_bh",
        )
        == RejectionResult.NOT_REJECTED
    )


@flaky(max_runs=2)
def test_given_non_linear_data_and_incorrect_dag_when_refute_invertible_model_then_reject_model():
    data = _generate_simple_non_linear_data()

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("Z", "Y"), ("Y", "X")]))  # X<-Y<-Z
    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)

    fit(causal_model, data)

    assert (
        refute_invertible_model(
            causal_model, data, independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=5)
        )
        == RejectionResult.REJECTED
    )
    assert (
        refute_invertible_model(
            causal_model,
            data,
            independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=5),
            fdr_control_method="fdr_bh",
        )
        == RejectionResult.REJECTED
    )


@flaky(max_runs=3)
def test_given_non_linear_data_and_incorrect_dag_with_collider_when_refute_invertible_model_then_reject_model():
    data = _generate_simple_non_linear_data()
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X", "Y"), ("Z", "Y")]))  # X->Y<-Z
    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)

    fit(causal_model, data)

    assert (
        refute_invertible_model(
            causal_model, data, independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=10)
        )
        == RejectionResult.REJECTED
    )
    assert (
        refute_invertible_model(
            causal_model,
            data,
            independence_test=lambda x, y: kernel_based(x, y, bootstrap_num_runs=10),
            fdr_control_method="fdr_bh",
        )
        == RejectionResult.REJECTED
    )
