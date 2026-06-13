"""Tests that CausalEstimator (and its subclasses) survive pickle and deepcopy.

Regression test for https://github.com/py-why/dowhy/issues/953 where
joblib's internal use of copy.deepcopy() inside get_new_estimator_object()
raised PicklingError: logger cannot be pickled on Python < 3.12.
"""

import copy
import pickle

import numpy as np
import pytest

import dowhy.datasets
from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.graph import build_graph_from_str


@pytest.fixture
def linear_estimate():
    """Fit a LinearRegressionEstimator and return (estimator, estimate)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        dataset = dowhy.datasets.linear_dataset(
            beta=1.0,
            num_common_causes=2,
            num_instruments=0,
            num_samples=200,
            treatment_is_binary=True,
        )
    finally:
        np.random.set_state(rng_state)
    graph = build_graph_from_str(dataset["gml_graph"])
    identified_estimand = identify_effect_auto(
        graph,
        action_nodes=dataset["treatment_name"],
        outcome_nodes=dataset["outcome_name"],
        observed_nodes=list(dataset["df"].columns),
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )
    estimator = LinearRegressionEstimator(
        identified_estimand=identified_estimand,
        test_significance=False,
    )
    estimator.fit(
        dataset["df"],
        effect_modifier_names=[],
    )
    estimate = estimator.estimate_effect(
        dataset["df"],
        control_value=0,
        treatment_value=1,
    )
    return estimator, estimate


class TestEstimatorPickling:
    """Verify that CausalEstimator subclasses can be pickled and deep-copied."""

    def test_pickle_round_trip(self, linear_estimate):
        """Pickle/unpickle must preserve the estimator's value and recreate the logger."""
        estimator, _ = linear_estimate
        data = pickle.dumps(estimator)
        restored = pickle.loads(data)
        assert hasattr(restored, "logger"), "logger must be recreated after unpickling"
        assert hasattr(restored, "_target_estimand")

    def test_deepcopy(self, linear_estimate):
        """copy.deepcopy() is used internally by get_new_estimator_object(); it must succeed."""
        estimator, _ = linear_estimate
        copied = copy.deepcopy(estimator)
        assert hasattr(copied, "logger"), "logger must be recreated after deepcopy"
        assert hasattr(copied, "_target_estimand")

    def test_get_new_estimator_object(self, linear_estimate):
        """get_new_estimator_object() calls deepcopy internally; the returned object must be usable."""
        estimator, estimate = linear_estimate
        new_estimator = estimator.get_new_estimator_object(estimator._target_estimand)
        assert new_estimator is not estimator
        assert hasattr(new_estimator, "logger"), "logger must exist on newly created estimator"
