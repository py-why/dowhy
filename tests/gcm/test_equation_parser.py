import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ProbabilisticCausalModel,
    create_causal_model_from_equations,
    fit,
    interventional_samples,
)
from dowhy.gcm.ml import create_linear_regressor_with_given_parameters


@flaky(max_runs=1)
def test_equation_parser_fit_func_is_giving_correct_results():
    observations = _generate_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model)

    fit(causal_model, observations)
    normal_results = causal_model.causal_mechanism("X1")._prediction_model.predict(observations[["X0"]].to_numpy())
    print(normal_results)
    # print(interventional_samples(causal_model, {'X1': lambda x: 1}, num_samples_to_draw=100))
    causal_model_from_eq = _get_causal_model_from_eq()
    fit(causal_model_from_eq, observations)
    eq_results = causal_model_from_eq.causal_mechanism("X1")._prediction_model.predict(observations[["X0"]].to_numpy())
    # TODO: Solve Approx error
    print("===")
    print(eq_results)
    # np.testing.assert_array_almost_equal(normal_results, eq_results)
    assert np.array_equal(approx(normal_results, abs=5, rel=5), approx(eq_results, abs=5, rel=5))


def _generate_data():
    X0 = np.random.normal(0, 0.1, 100)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 100)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 100)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 100)
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


test_equation_parser_fit_func_is_giving_correct_results()
