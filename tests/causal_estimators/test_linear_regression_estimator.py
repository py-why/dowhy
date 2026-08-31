import numpy as np
import pytest
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

    def test_none_identifier_method_does_not_raise(self):
        """identifier_method=None (functional API) should not raise ValueError."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        # functional API leaves identifier_method=None; estimator should not raise
        target_estimand.identifier_method = None
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        estimator.fit(data["df"])  # should not raise

    def test_test_significance_returns_scalar_float_for_single_treatment(self):
        """_test_significance should return a plain float p-value for a single treatment variable.

        Regression test for https://github.com/py-why/dowhy/issues/1019 where the p-value
        was returned as a 1-element numpy array (``array([0.])``) instead of a scalar float.
        """
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_samples=500,
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
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand, test_significance=True)
        estimator.fit(data["df"])
        ate_estimate = estimator.estimate_effect(data["df"], control_value=0, treatment_value=1)
        signif = ate_estimate.test_stat_significance()
        p_value = signif["p_value"]
        assert isinstance(
            p_value, float
        ), f"Expected scalar float p-value for single treatment, got {type(p_value)}: {p_value!r}"
        assert 0.0 <= p_value <= 1.0, f"p-value {p_value} is not in [0, 1]"

    def test_test_significance_returns_array_for_multiple_treatments(self):
        """_test_significance should return a numpy array of p-values for multiple treatment variables."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=2,
            num_samples=500,
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
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand, test_significance=True)
        estimator.fit(data["df"])
        ate_estimate = estimator.estimate_effect(data["df"], control_value=0, treatment_value=1)
        signif = ate_estimate.test_stat_significance()
        p_value = signif["p_value"]
        assert isinstance(
            p_value, np.ndarray
        ), f"Expected numpy array p-value for multiple treatments, got {type(p_value)}: {p_value!r}"
        assert p_value.shape == (2,), f"Expected shape (2,), got {p_value.shape}"
        assert np.all((p_value >= 0.0) & (p_value <= 1.0)), f"p-values {p_value} are not all in [0, 1]"

    @mark.parametrize("invalid_method", ["frontdoor", "iv", "mediation"])
    def test_invalid_identifier_method_raises(self, invalid_method):
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True,
        )
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method(invalid_method)
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        with pytest.raises(ValueError, match="only supports backdoor and general_adjustment"):
            estimator.fit(data["df"])

    def test_evaluate_effect_strength_binary_treatment(self):
        """evaluate_effect_strength must not raise for a single binary treatment.

        Regression test for #416: estimate_effect_naive used data[list] instead of
        data[col_name], returning a DataFrame that caused `ValueError: Cannot index with
        multidimensional key` inside `.loc[]`.
        """
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_samples=500,
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
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        estimator.fit(data["df"])
        ate_estimate = estimator.estimate_effect(data["df"], control_value=0, treatment_value=1)
        # Should not raise ValueError: Cannot index with multidimensional key
        strength = estimator.evaluate_effect_strength(data["df"], ate_estimate)
        assert "fraction-effect" in strength
        assert np.isfinite(strength["fraction-effect"])

    def test_evaluate_effect_strength_non_binary_treatment(self):
        """estimate_effect_naive must respect actual treatment_value / control_value, not hardcoded 0/1.

        Regression test for #416: the old code used hardcoded ``== 1`` and ``== 0``, so
        non-binary treatments (e.g. control_value=1, treatment_value=2) would silently
        compute the wrong effect-strength ratio (selecting no rows).
        """
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=False,
        )
        df = data["df"]
        # Recode continuous treatment to {1, 2} so both control and treatment rows exist (non-standard values)
        df[data["treatment_name"][0]] = np.where(df[data["treatment_name"][0]] > 0, 2, 1)
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(df.columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        estimator.fit(df)
        ate_estimate = estimator.estimate_effect(df, control_value=1, treatment_value=2)
        # Should not raise; fraction-effect must be a finite number
        strength = estimator.evaluate_effect_strength(df, ate_estimate)
        assert np.isfinite(strength["fraction-effect"])

    # -------------------------------------------------------------------------
    # Tests for delta-method CI/SE with effect modifiers (issue #336)
    # -------------------------------------------------------------------------

    def _make_estimator_with_effect_modifiers(self, num_effect_modifiers=1, num_samples=2000):
        """Helper: return (df, estimator) for a linear dataset with effect modifiers."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_effect_modifiers=num_effect_modifiers,
            num_samples=num_samples,
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
        effect_modifier_names = data["effect_modifier_names"]
        estimator = LinearRegressionEstimator(
            identified_estimand=target_estimand,
            confidence_intervals=True,
        )
        estimator.fit(data["df"], effect_modifier_names=effect_modifier_names)
        return data["df"], estimator, data["effect_modifier_names"]

    def test_delta_method_ci_does_not_raise_with_effect_modifiers(self):
        """_estimate_confidence_intervals must not raise NotImplementedError when effect modifiers are present.

        Regression test for issue #336: the method previously raised NotImplementedError.
        """
        df, estimator, em_names = self._make_estimator_with_effect_modifiers()
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        # Must not raise; CI should be retrieved without bootstrapping
        ci = ate_estimate.get_confidence_intervals()
        assert ci is not None, "CI should not be None when confidence_intervals=True"

    def test_delta_method_ci_shape_with_one_effect_modifier(self):
        """CI array returned for the effect-modifier case must have shape (1, 2)."""
        df, estimator, em_names = self._make_estimator_with_effect_modifiers(num_effect_modifiers=1)
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        ci = ate_estimate.get_confidence_intervals()
        ci_arr = np.array(ci)
        assert ci_arr.shape == (1, 2), f"Expected shape (1, 2), got {ci_arr.shape}"

    def test_delta_method_ci_finite_and_ordered(self):
        """CI bounds must be finite and lower < upper."""
        df, estimator, em_names = self._make_estimator_with_effect_modifiers()
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        ci = np.array(ate_estimate.get_confidence_intervals())
        assert np.all(np.isfinite(ci)), f"CI contains non-finite values: {ci}"
        assert ci[0, 0] < ci[0, 1], f"Lower bound {ci[0, 0]} not less than upper bound {ci[0, 1]}"

    def test_delta_method_ci_contains_true_ate(self):
        """The 95% delta-method CI should contain the point estimate and be centred on it."""
        df, estimator, em_names = self._make_estimator_with_effect_modifiers(num_samples=5000)
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        ci = np.array(ate_estimate.get_confidence_intervals())
        point_estimate = ate_estimate.value
        # The CI is always centred on the ATE point estimate
        assert (
            ci[0, 0] <= point_estimate <= ci[0, 1]
        ), f"Point estimate ({point_estimate:.4f}) not contained in CI [{ci[0, 0]:.4f}, {ci[0, 1]:.4f}]"
        # With n=5000 and a low-noise linear DGP the CI should be narrow but not degenerate
        ci_width = ci[0, 1] - ci[0, 0]
        assert (
            0 < ci_width < point_estimate
        ), f"CI width {ci_width:.6f} is implausible for estimate {point_estimate:.4f}"

    def test_delta_method_se_does_not_raise_with_effect_modifiers(self):
        """_estimate_std_error must not raise NotImplementedError when effect modifiers are present.

        Regression test for issue #336.
        """
        df, estimator, em_names = self._make_estimator_with_effect_modifiers()
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        se = ate_estimate.get_standard_error()
        assert se is not None
        assert np.isfinite(se).all(), f"SE contains non-finite values: {se}"
        assert (np.array(se) > 0).all(), f"SE must be positive, got {se}"

    def test_delta_method_ci_with_multiple_effect_modifiers(self):
        """Delta-method CI must work when there are multiple effect modifiers."""
        df, estimator, em_names = self._make_estimator_with_effect_modifiers(num_effect_modifiers=2)
        ate_estimate = estimator.estimate_effect(df, control_value=0, treatment_value=1)
        ci = np.array(ate_estimate.get_confidence_intervals())
        assert ci.shape == (1, 2)
        assert np.all(np.isfinite(ci))
        assert ci[0, 0] < ci[0, 1]
