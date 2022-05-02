import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import AdditiveNoiseModel, distribution_change, distribution_change_of_graphs, fit, \
    ProbabilisticCausalModel, EmpiricalDistribution
from dowhy.gcm.ml import create_linear_regressor
from dowhy.gcm.shapley import ShapleyConfig


@flaky(max_runs=5)
def test_distribution_change():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    original_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 3 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    _assign_causal_mechanisms(causal_model)

    results = distribution_change(causal_model,
                                  original_observations,
                                  outlier_observations,
                                  'X3',
                                  shapley_config=ShapleyConfig(n_jobs=1))

    assert results['X3'] > results['X2']
    assert results['X2'] > results['X0']
    assert 'X1' not in results
    assert results['X0'] == approx(0, abs=0.15)


@flaky(max_runs=5)
def test_distribution_change_of_graphs():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    original_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 3 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    causal_model_old = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    _assign_causal_mechanisms(causal_model_old)
    causal_model_new = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    _assign_causal_mechanisms(causal_model_new)

    fit(causal_model_old, original_observations)
    fit(causal_model_new, outlier_observations)

    results = distribution_change_of_graphs(causal_model_old, causal_model_new, 'X3')

    assert results['X3'] > results['X2'] > results['X0']
    assert 'X1' not in results
    assert results['X0'] == approx(0, abs=0.1)


@flaky(max_runs=5)
def test_when_using_distribution_change_without_fdrc_then_returns_valid_results():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    original_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 3 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    _assign_causal_mechanisms(causal_model)

    results = distribution_change(causal_model,
                                  original_observations,
                                  outlier_observations,
                                  'X3',
                                  mechanism_change_test_fdr_control_method=None)

    assert results['X3'] > results['X2'] > results['X0']
    assert 'X1' not in results


@flaky(max_runs=5)
def test_when_using_distribution_change_with_return_additional_info_then_returns_additional_info():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 3 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    _assign_causal_mechanisms(causal_model)

    attributions, mechanism_changes, causal_model_old, causal_model_new = distribution_change(
        causal_model,
        original_observations,
        outlier_observations,
        'X3',
        mechanism_change_test_fdr_control_method=None,
        return_additional_info=True)

    assert attributions['X3'] > attributions['X2'] > attributions['X0']
    assert 'X1' not in attributions

    assert not mechanism_changes['X0']
    assert mechanism_changes['X2']
    assert mechanism_changes['X3']

    assert 'X1' not in causal_model_old.graph.nodes
    assert 'X1' not in causal_model_new.graph.nodes

    for node in causal_model.graph.nodes:
        if node == 'X1':
            continue
        assert type(causal_model_old.causal_mechanism(node)) == type(causal_model_new.causal_mechanism(node))


def _assign_causal_mechanisms(causal_model):
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(create_linear_regressor()))
