from pytest import mark

import dowhy.datasets
from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreStratificationEstimator(object):
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
                0.1,
                PropensityScoreStratificationEstimator,
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
                0.1,
                PropensityScoreStratificationEstimator,
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
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )

    @mark.usefixtures("fixed_seed")
    def test_nondefault_treatment_control_values(self):
        """Regression test: treatment_value/control_value must be respected.

        Before this fix, _get_strata() hardcoded ``== 1`` / ``== 0`` instead of
        using ``_treatment_value`` / ``_control_value``, so the estimate was
        wrong whenever the caller used reversed or non-standard values.
        """
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_samples=10000,
            treatment_is_binary=True,
        )
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")

        def _estimate(tv, cv):
            est = PropensityScoreStratificationEstimator(
                identified_estimand=target_estimand,
                num_strata=5,
                treatment_value=tv,
                control_value=cv,
            )
            est.fit(data["df"], effect_modifier_names=data["effect_modifier_names"])
            return est.estimate_effect(
                data["df"],
                treatment_value=tv,
                control_value=cv,
                target_units="ate",
            ).value

        forward = _estimate(tv=1, cv=0)
        reversed_ = _estimate(tv=0, cv=1)

        # Reversing treatment and control should negate the ATE estimate.
        assert abs(forward + reversed_) < 0.3 * abs(forward), (
            f"Expected reversed ATE ({reversed_:.4f}) ≈ -{forward:.4f}, "
            f"but difference ({abs(forward + reversed_):.4f}) is too large"
        )
