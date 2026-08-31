"""Tests for dowhy.gcm.fitting_sampling – fit, fit_causal_model_of_target, draw_samples."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel
from dowhy.gcm.causal_models import StructuralCausalModel
from dowhy.gcm.fitting_sampling import draw_samples, fit, fit_causal_model_of_target
from dowhy.gcm.ml import create_linear_regressor
from dowhy.gcm.stochastic_models import EmpiricalDistribution


def _make_simple_model() -> tuple:
    """Return a (causal_model, data) pair with X → Y."""
    np.random.seed(0)
    causal_model = StructuralCausalModel(nx.DiGraph([("X", "Y")]))
    causal_model.set_causal_mechanism("X", EmpiricalDistribution())
    causal_model.set_causal_mechanism("Y", AdditiveNoiseModel(create_linear_regressor()))
    data = pd.DataFrame({"X": np.random.randn(200), "Y": np.random.randn(200)})
    return causal_model, data


def test_when_fitting_model_then_succeeds_without_error():
    causal_model, data = _make_simple_model()
    fit(causal_model, data)  # must not raise


def test_when_fitting_model_with_missing_data_column_then_raises_runtime_error():
    causal_model, data = _make_simple_model()
    fit(causal_model, data)
    with pytest.raises(RuntimeError, match="Could not find data for node Y"):
        fit(causal_model, data[["X"]])  # drop the Y column


def test_when_fitting_causal_model_of_target_with_nan_values_then_ignores_nan_rows():
    """fit_causal_model_of_target masks out NaN rows in the target column."""
    causal_model, data = _make_simple_model()
    fit(causal_model, data)  # prime the X mechanism first

    data_with_nans = data.copy()
    data_with_nans.loc[0:9, "Y"] = np.nan  # inject 10 NaN values

    # Must not raise despite NaNs in Y
    fit_causal_model_of_target(causal_model, "Y", data_with_nans)


def test_when_fitting_causal_model_of_root_node_with_nan_values_then_ignores_nan_rows():
    """fit_causal_model_of_target also masks NaNs for root nodes."""
    causal_model, data = _make_simple_model()

    data_with_nans = data.copy()
    data_with_nans.loc[0:4, "X"] = np.nan

    # Must not raise despite NaNs in X (root node)
    fit_causal_model_of_target(causal_model, "X", data_with_nans)


def test_when_drawing_samples_then_returns_correct_shape_and_column_names():
    causal_model, data = _make_simple_model()
    fit(causal_model, data)
    samples = draw_samples(causal_model, num_samples=50)
    assert samples.shape == (50, 2)
    assert set(samples.columns) == {"X", "Y"}


def test_when_drawing_samples_then_returns_dataframe():
    causal_model, data = _make_simple_model()
    fit(causal_model, data)
    samples = draw_samples(causal_model, num_samples=10)
    assert isinstance(samples, pd.DataFrame)


def test_when_drawing_samples_then_no_nan_values():
    causal_model, data = _make_simple_model()
    fit(causal_model, data)
    samples = draw_samples(causal_model, num_samples=100)
    assert not samples.isnull().any().any()


def test_when_drawing_samples_from_chain_graph_then_correct_columns_returned():
    """X → Y → Z chain: draw_samples must return all three nodes."""
    np.random.seed(1)
    causal_model = StructuralCausalModel(nx.DiGraph([("X", "Y"), ("Y", "Z")]))
    causal_model.set_causal_mechanism("X", EmpiricalDistribution())
    causal_model.set_causal_mechanism("Y", AdditiveNoiseModel(create_linear_regressor()))
    causal_model.set_causal_mechanism("Z", AdditiveNoiseModel(create_linear_regressor()))
    data = pd.DataFrame(
        {
            "X": np.random.randn(200),
            "Y": np.random.randn(200),
            "Z": np.random.randn(200),
        }
    )
    fit(causal_model, data)
    samples = draw_samples(causal_model, num_samples=30)
    assert set(samples.columns) == {"X", "Y", "Z"}
    assert samples.shape == (30, 3)
