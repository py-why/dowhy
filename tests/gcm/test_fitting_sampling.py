"""Tests for dowhy.gcm.fitting_sampling (fit, fit_causal_model_of_target, draw_samples)."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution, ProbabilisticCausalModel, draw_samples, fit
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target
from dowhy.gcm.ml import create_linear_regressor


def _simple_scm():
    """Return a minimal SCM: X0 → X1, with assigned causal mechanisms."""
    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1")]))
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    return scm


def _simple_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    x1 = 2 * x0 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"X0": x0, "X1": x1})


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_raises_error_when_node_missing_from_data():
    scm = _simple_scm()
    data = _simple_data()
    data_missing_col = data.drop(columns=["X1"])
    with pytest.raises(RuntimeError, match="X1"):
        fit(scm, data_missing_col)


def test_fit_completes_without_error_for_valid_data():
    scm = _simple_scm()
    fit(scm, _simple_data())


def test_fit_stores_parents_during_fit_for_non_root_node():
    scm = _simple_scm()
    fit(scm, _simple_data())
    assert scm.graph.nodes["X1"][PARENTS_DURING_FIT] == ["X0"]


def test_fit_stores_empty_parents_during_fit_for_root_node():
    scm = _simple_scm()
    fit(scm, _simple_data())
    assert scm.graph.nodes["X0"][PARENTS_DURING_FIT] == []


def test_fit_skips_nan_rows_in_target_variable():
    """Rows where the TARGET has NaN should be silently dropped during fit."""
    scm = _simple_scm()
    data = _simple_data(n=300, seed=42)
    # Introduce NaN in X1 (the non-root target)
    data_with_nan = data.copy()
    data_with_nan.loc[[10, 20, 30], "X1"] = np.nan
    # Fit should succeed without error despite NaN in target
    fit(scm, data_with_nan)


def test_fit_returns_none_when_return_evaluation_summary_is_false():
    scm = _simple_scm()
    result = fit(scm, _simple_data())
    assert result is None


# ---------------------------------------------------------------------------
# fit_causal_model_of_target()
# ---------------------------------------------------------------------------


def test_fit_causal_model_of_target_root_node_learns_distribution():
    scm = _simple_scm()
    data = _simple_data(n=500, seed=1)
    fit_causal_model_of_target(scm, "X0", data)
    assert PARENTS_DURING_FIT in scm.graph.nodes["X0"]
    assert scm.graph.nodes["X0"][PARENTS_DURING_FIT] == []


def test_fit_causal_model_of_target_non_root_node():
    scm = _simple_scm()
    data = _simple_data(n=500, seed=2)
    fit_causal_model_of_target(scm, "X1", data)
    assert scm.graph.nodes["X1"][PARENTS_DURING_FIT] == ["X0"]


def test_fit_causal_model_of_target_raises_when_node_has_no_mechanism():
    graph = nx.DiGraph([("X0", "X1")])
    scm = ProbabilisticCausalModel(graph)
    # Only assign X0 mechanism; X1 has none
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    data = _simple_data()
    with pytest.raises(ValueError, match="X1"):
        fit_causal_model_of_target(scm, "X1", data)


def test_fit_causal_model_of_target_raises_when_wrong_model_type_on_root():
    """Root node must have a StochasticModel; using a ConditionalStochasticModel should raise."""
    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1")]))
    # Assign a ConditionalStochasticModel (AdditiveNoiseModel) to root node X0 — invalid
    scm.set_causal_mechanism("X0", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    data = _simple_data()
    with pytest.raises(RuntimeError):
        fit_causal_model_of_target(scm, "X0", data)


# ---------------------------------------------------------------------------
# draw_samples()
# ---------------------------------------------------------------------------


def test_draw_samples_returns_correct_shape():
    scm = _simple_scm()
    fit(scm, _simple_data())
    samples = draw_samples(scm, num_samples=100)
    assert samples.shape == (100, 2)
    assert set(samples.columns) == {"X0", "X1"}


def test_draw_samples_three_node_chain_correct_column_count():
    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2")]))
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, 200)
    x1 = 2 * x0 + rng.normal(0, 0.1, 200)
    x2 = 0.5 * x1 + rng.normal(0, 0.1, 200)
    fit(scm, pd.DataFrame({"X0": x0, "X1": x1, "X2": x2}))
    samples = draw_samples(scm, num_samples=50)
    assert samples.shape == (50, 3)


def test_draw_samples_preserves_node_names():
    scm = _simple_scm()
    fit(scm, _simple_data())
    samples = draw_samples(scm, num_samples=10)
    assert "X0" in samples.columns
    assert "X1" in samples.columns
