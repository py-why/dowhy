import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ProbabilisticCausalModel,
    draw_samples,
    fit,
    is_root_node,
)
from dowhy.gcm.ml import create_linear_regressor


@flaky(max_runs=2)
def test_fit_and_draw_samples():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X3", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(scm, original_observations)

    generated_samples = draw_samples(scm, 1000)

    assert np.mean(generated_samples["X0"]) == approx(np.mean(X0), abs=0.1)
    assert np.std(generated_samples["X0"]) == approx(np.std(X0), abs=0.2)
    assert np.mean(generated_samples["X1"]) == approx(np.mean(X1), abs=0.1)
    assert np.std(generated_samples["X1"]) == approx(np.std(X1), abs=0.2)
    assert np.mean(generated_samples["X2"]) == approx(np.mean(X2), abs=0.1)
    assert np.std(generated_samples["X2"]) == approx(np.std(X2), abs=0.2)
    assert np.mean(generated_samples["X3"]) == approx(np.mean(X3), abs=0.1)
    assert np.std(generated_samples["X3"]) == approx(np.std(X3), abs=0.2)


def test_set_causal_model_raises_error():
    with pytest.raises(ValueError):
        ProbabilisticCausalModel().set_causal_mechanism("X0", EmpiricalDistribution())


def test_is_root_node():
    graph = nx.DiGraph([("X", "Z"), ("Y", "Z")])
    assert is_root_node(graph, "X") == True
    assert is_root_node(graph, "Y") == True
    assert is_root_node(graph, "Z") == False
