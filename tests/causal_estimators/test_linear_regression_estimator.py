import numpy as np
from pytest import mark

import dowhy.datasets
from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator, TestGraphObject, example_graph


@mark.usefixtures("fixed_seed")
class TestLinearRegressionEstimator(object):
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
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
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
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
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
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
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
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    True,
                ],
                [
                    False,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    False,
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

    def test_general_adjustment_estimation_on_example_graphs(self, example_graph: TestGraphObject):
        data = dowhy.datasets.linear_dataset_from_graph(
            example_graph.graph,
            example_graph.action_nodes,
            example_graph.outcome_node,
            treatments_are_binary=True,
            outcome_is_binary=False,
            num_samples=50000,
        )
        data["df"] = data["df"][example_graph.observed_nodes]
        estimator_tester = SimpleEstimator(0.1, LinearRegressionEstimator, identifier_method="general_adjustment")
        estimator_tester.custom_data_average_treatment_effect_test(data)


class TestLinearRegressionAsymptoticCI:
    """Tests for the Delta-method asymptotic CI/SE with effect modifiers (issue #336)."""

    def _make_dataset_and_estimand(self, num_effect_modifiers=1, num_common_causes=1, num_treatments=1, seed=42):
        np.random.seed(seed)
        data = dowhy.datasets.linear_dataset(
            beta=5,
            num_common_causes=num_common_causes,
            num_instruments=0,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            num_samples=2000,
            treatment_is_binary=False,
        )
        gml_graph = data["gml_graph"]
        df = data["df"]
        target_estimand = identify_effect_auto(
            build_graph_from_str(gml_graph),
            observed_nodes=list(df.columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")
        return data, target_estimand

    def test_ci_returned_not_raises_single_treatment_single_em(self):
        """No NotImplementedError for single treatment + single effect modifier."""
        data, estimand = self._make_dataset_and_estimand(num_effect_modifiers=1)
        estimator = LinearRegressionEstimator(
            identified_estimand=estimand,
            confidence_intervals=True,
        )
        estimator.fit(data["df"], effect_modifier_names=data["effect_modifier_names"])
        estimate = estimator.estimate_effect(
            data["df"],
            treatment_value=1,
            control_value=0,
            confidence_intervals=True,
        )
        ci = estimate.get_confidence_intervals()
        assert ci is not None
        assert ci.shape == (1, 2), f"Expected shape (1,2), got {ci.shape}"
        lower, upper = ci[0]
        assert lower < upper, "CI lower bound must be less than upper bound"

    def test_ci_contains_true_ate_with_high_probability(self):
        """95% CI should bracket the true ATE on a large sample."""
        data, estimand = self._make_dataset_and_estimand(num_effect_modifiers=2, num_common_causes=1, seed=0)
        estimator = LinearRegressionEstimator(
            identified_estimand=estimand,
            confidence_intervals=True,
            confidence_level=0.95,
        )
        estimator.fit(data["df"], effect_modifier_names=data["effect_modifier_names"])
        estimate = estimator.estimate_effect(
            data["df"],
            treatment_value=1,
            control_value=0,
            confidence_intervals=True,
        )
        ci = estimate.get_confidence_intervals()
        lower, upper = ci[0]
        true_ate = data["ate"]
        assert lower <= true_ate <= upper, f"True ATE {true_ate:.4f} not inside 95% CI [{lower:.4f}, {upper:.4f}]"

    def test_std_error_positive_with_effect_modifier(self):
        """Standard error should be positive and finite when effect modifiers are present."""
        data, estimand = self._make_dataset_and_estimand(num_effect_modifiers=1)
        estimator = LinearRegressionEstimator(
            identified_estimand=estimand,
            test_significance=True,
            confidence_intervals=True,
        )
        estimator.fit(data["df"], effect_modifier_names=data["effect_modifier_names"])
        estimate = estimator.estimate_effect(
            data["df"],
            treatment_value=1,
            control_value=0,
            confidence_intervals=True,
        )
        se = estimate.get_standard_error()
        assert se is not None
        assert np.all(np.isfinite(se)), "SE should be finite"
        assert np.all(se > 0), "SE should be positive"

    def test_ci_consistent_with_no_effect_modifier(self):
        """With no effect modifiers, Delta-method and direct statsmodels CI should agree."""
        data, estimand = self._make_dataset_and_estimand(num_effect_modifiers=0)
        estimator = LinearRegressionEstimator(
            identified_estimand=estimand,
            confidence_intervals=True,
            confidence_level=0.95,
        )
        estimator.fit(data["df"], effect_modifier_names=[])
        estimate = estimator.estimate_effect(
            data["df"],
            treatment_value=1,
            control_value=0,
            confidence_intervals=True,
        )
        ci = estimate.get_confidence_intervals()
        assert ci is not None
        lower, upper = ci[0]
        assert lower < upper
