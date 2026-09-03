import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.doubly_robust_estimator import DoublyRobustEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator


def _make_dr_data(n=2000, seed=0):
    """Return a simple confounded dataset and its GML graph string."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    T = (X + rng.normal(size=n) > 0).astype(int)
    Y = 2 * T + X + rng.normal(size=n)
    df = pd.DataFrame({"X": X, "T": T, "Y": Y})
    gml = 'graph [directed 1 node [id "T" label "T"] node [id "Y" label "Y"] node [id "X" label "X"] edge [source "X" target "T"] edge [source "X" target "Y"] edge [source "T" target "Y"]]'  # noqa: E501
    return df, gml


def _fit_dr_estimator(df, gml, effect_modifier_names=None):
    """Helper: identify, construct, and fit a DoublyRobustEstimator."""
    estimand = identify_effect_auto(
        build_graph_from_str(gml),
        observed_nodes=list(df.columns),
        action_nodes=["T"],
        outcome_nodes=["Y"],
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )
    estimand.set_identifier_method("backdoor")
    estimator = DoublyRobustEstimator(
        identified_estimand=estimand,
        num_simulations=10,
        num_null_simulations=10,
    )
    estimator.fit(df, effect_modifier_names=effect_modifier_names or [])
    return estimator


@mark.usefixtures("fixed_seed")
class TestDoublyRobustEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "treatment_is_category",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.1,
                DoublyRobustEstimator,
                [1, 2],
                [0, 1],
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
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.2,
                DoublyRobustEstimator,
                [1, 2],
                [
                    0,
                ],
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
                [
                    True,
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
        treatment_is_category,
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
            treatment_is_category=treatment_is_category,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )

    def test_multiple_treatments_raises_value_error(self):
        estimator_tester = SimpleEstimator(error_tolerance=0.5, Estimator=DoublyRobustEstimator)
        with pytest.raises(ValueError, match="cannot handle more than one treatment variable"):
            estimator_tester.average_treatment_effect_testsuite(
                num_common_causes=[1],
                num_instruments=[0],
                num_effect_modifiers=[0],
                num_treatments=[2],
                treatment_is_binary=[True],
                outcome_is_binary=[False],
                confidence_intervals=[False],
                test_significance=[False],
                method_params={"num_simulations": 10, "num_null_simulations": 10},
            )

    def test_effect_modifiers_raises_not_implemented_error(self):
        """fit() must raise NotImplementedError when effect modifiers are provided."""
        df, gml = _make_dr_data()
        # Add an effect modifier column
        df["EM"] = df["X"] ** 2
        gml_em = gml.replace(
            'edge [source "T" target "Y"]',
            'node [id "EM" label "EM"] edge [source "T" target "Y"] edge [source "EM" target "Y"]',
        )
        estimand = identify_effect_auto(
            build_graph_from_str(gml_em),
            observed_nodes=list(df.columns),
            action_nodes=["T"],
            outcome_nodes=["Y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        estimand.set_identifier_method("backdoor")
        estimator = DoublyRobustEstimator(
            identified_estimand=estimand,
            num_simulations=10,
            num_null_simulations=10,
        )
        with pytest.raises(NotImplementedError, match="Effect Modifiers not supported"):
            estimator.fit(df, effect_modifier_names=["EM"])

    def test_target_units_att_raises_not_implemented_error(self):
        """estimate_effect() must raise NotImplementedError for target_units other than 'ate'."""
        df, gml = _make_dr_data()
        estimator = _fit_dr_estimator(df, gml)
        with pytest.raises(NotImplementedError, match="ATE is the only target unit"):
            estimator.estimate_effect(df, target_units="att")

    def test_target_units_atc_raises_not_implemented_error(self):
        """estimate_effect() must raise NotImplementedError for target_units='atc'."""
        df, gml = _make_dr_data()
        estimator = _fit_dr_estimator(df, gml)
        with pytest.raises(NotImplementedError, match="ATE is the only target unit"):
            estimator.estimate_effect(df, target_units="atc")
