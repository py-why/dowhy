import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ProbabilisticCausalModel,
    create_causal_model_from_equations,
    fit,
)
from dowhy.gcm.causal_mechanisms import ConditionalStochasticModel
from dowhy.gcm.ml import create_linear_regressor_with_given_parameters


@flaky(max_runs=2)
def test_equation_parser_fit_func_is_giving_correct_results():
    observations = _generate_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model)

    fit(causal_model, observations)
    normal_results = causal_model.causal_mechanism("X1")._prediction_model.predict(observations[["X0"]].to_numpy())
    normal_results = np.around(normal_results, 2)
    causal_model_from_eq = _get_causal_model_from_eq()
    fit(causal_model_from_eq, observations)
    eq_results = causal_model_from_eq.causal_mechanism("X1")._prediction_model.predict(observations[["X0"]].to_numpy())
    eq_results = np.around(eq_results, 2)
    assert np.array_equal(normal_results, eq_results)


def test_variables_are_sorted_alphabetically_in_custom_predict_method():
    causal_model = create_causal_model_from_equations(
        """
    A = norm(loc=0,scale=0.1)
    B = norm(loc=0, scale=0.1)
    Y = 0.5*B + 2*A+ norm(loc=0, scale=0.1)
    """
    )
    A = np.random.normal(0, 0.1, 10)
    B = np.random.normal(0, 0.1, 10)
    Y = 0.5 * B + 2 * A

    observations = pd.DataFrame({"A": A, "B": B, "Y": Y})
    eq_results = causal_model.causal_mechanism("Y")._prediction_model.predict(observations[["A", "B"]].to_numpy())
    assert np.array_equal(np.around(Y, 2), np.around(eq_results.ravel(), 2))


def test_unknown_causal_model_relationship_is_undefined():
    causal_model = create_causal_model_from_equations(
        """
    A = norm(loc=0,scale=0.1)
    B = norm(loc=0, scale=0.1)
    Y = 0.5*B + 2*A+ norm(loc=0, scale=0.1)
    Z->Y,A
    """
    )
    assert "Z" in causal_model.graph.nodes
    try:
        mech = causal_model.causal_mechanism("Z")
        raise AssertionError("The causal mechanism is defined for unknown model node!")
    except KeyError as ke:
        pass


def test_known_causal_model_node_is_correctly_identified():
    causal_model = create_causal_model_from_equations(
        """
    A = norm(loc=0,scale=0.1)
    B = norm(loc=0, scale=0.1)
    Y = 0.5*B + 2*A+ norm(loc=0, scale=0.1)
    Z->Y,A
    C = exp(A) + 5 * Z + parametric()
    """
    )
    list_of_nodes = {"A", "B", "C", "Z", "Y"}
    list_of_nodes_from_graph = set(causal_model.graph.nodes)
    assert list_of_nodes.issubset(list_of_nodes_from_graph) and list_of_nodes_from_graph.issubset(list_of_nodes)
    assert isinstance(causal_model.causal_mechanism("C"), ConditionalStochasticModel)


def _generate_data():
    X0 = np.random.normal(0, 0.1, 100)
    X1 = 2 * X0
    X2 = 0.5 * X0
    X3 = 0.5 * X2
    observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    return observations


def _get_causal_model_from_eq():
    causal_model = create_causal_model_from_equations(
        """
    X0 = norm(loc=0,scale=0.1)
    X1 = 2*X0 + norm(loc=0, scale=0.1)
    X2 = 0.5*X0 + norm(loc=0, scale=0.1)
    X3 = 0.5*X2 + norm(loc=0, scale=0.1)
    """
    )
    return causal_model


def _assign_causal_mechanisms(causal_model):
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism(
        "X1", AdditiveNoiseModel(create_linear_regressor_with_given_parameters(coefficients=np.array([2])))
    )
    causal_model.set_causal_mechanism(
        "X2", AdditiveNoiseModel(create_linear_regressor_with_given_parameters(coefficients=np.array([0.5])))
    )
    causal_model.set_causal_mechanism(
        "X3", AdditiveNoiseModel(create_linear_regressor_with_given_parameters(coefficients=np.array([0.5])))
    )
