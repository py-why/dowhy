import numpy as np
import pandas as pd
from pytest import mark

from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.propensity_score_matching_estimator import PropensityScoreMatchingEstimator
from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreWeightingEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.4,
                PropensityScoreWeightingEstimator,
                [1, 2],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.4,
                PropensityScoreWeightingEstimator,
                [1, 2],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
        ],
    )
    def test_average_treatment_effect(
        self,
        error_tolerance,
        Estimator,
        num_common_causes,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        treatment_is_binary,
        outcome_is_binary,
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 1, "num_null_simulations": 1},
        )


def _make_simple_binary_dataset():
    """Create a minimal dataset with a binary treatment and one common cause."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(int)
    y = 0.5 * t + x + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def _make_identified_estimand(data):
    gml_graph = (
        "graph [directed 1"
        " node [id 0 label 'x'] node [id 1 label 't'] node [id 2 label 'y']"
        " edge [source 0 target 1] edge [source 0 target 2] edge [source 1 target 2]]"
    )
    graph = build_graph_from_str(gml_graph)
    estimand = identify_effect_auto(
        graph,
        observed_nodes=list(data.columns),
        action_nodes=["t"],
        outcome_nodes=["y"],
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )
    estimand.set_identifier_method("backdoor")
    return estimand


class TestPropensityScoreEstimatorsDefaultTargetUnits:
    """Regression tests for the target_units="ate" default (was None, causing ValueError)."""

    def test_psw_estimate_effect_default_target_units_does_not_raise(self):
        data = _make_simple_binary_dataset()
        estimand = _make_identified_estimand(data)
        estimator = PropensityScoreWeightingEstimator(identified_estimand=estimand)
        estimator.fit(data)
        # Calling without target_units must not raise ValueError.
        result = estimator.estimate_effect(data)
        assert np.isfinite(result.value), "ATE should be a finite number"

    def test_psm_estimate_effect_default_target_units_does_not_raise(self):
        data = _make_simple_binary_dataset()
        estimand = _make_identified_estimand(data)
        estimator = PropensityScoreMatchingEstimator(identified_estimand=estimand)
        estimator.fit(data)
        result = estimator.estimate_effect(data)
        assert np.isfinite(result.value), "ATE should be a finite number"

    def test_pss_estimate_effect_default_target_units_does_not_raise(self):
        data = _make_simple_binary_dataset()
        estimand = _make_identified_estimand(data)
        estimator = PropensityScoreStratificationEstimator(identified_estimand=estimand)
        estimator.fit(data)
        result = estimator.estimate_effect(data)
        assert np.isfinite(result.value), "ATE should be a finite number"
