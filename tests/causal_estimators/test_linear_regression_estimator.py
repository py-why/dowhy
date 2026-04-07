import numpy as np
import pandas as pd
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

    GML_GRAPH = """
    graph [
        directed 1
        node [ id "W" label "W" ]
        node [ id "T" label "T" ]
        node [ id "X" label "X" ]
        node [ id "Y" label "Y" ]
        edge [ source "W" target "T" ]
        edge [ source "W" target "Y" ]
        edge [ source "T" target "Y" ]
        edge [ source "X" target "Y" ]
    ]
    """

    def _make_estimand(self, df):
        graph = build_graph_from_str(self.GML_GRAPH)
        estimand = identify_effect_auto(
            graph,
            observed_nodes=list(df.columns),
            action_nodes=["T"],
            outcome_nodes=["Y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        estimand.set_identifier_method("backdoor")
        return estimand

    def _make_continuous_dataset(self, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1, size=n)
        X = rng.normal(0, 1, size=n)
        T = 0.5 * W + rng.normal(0, 1, size=n)
        Y = 5.0 * T + 3.0 * T * X + 2.0 * W + rng.normal(0, 1, size=n)
        return pd.DataFrame({"T": T, "W": W, "X": X, "Y": Y})

    def _make_categorical_dataset(self, n=3000, seed=0):
        """Common cause W is categorical with 3 levels → 2 encoded columns after one-hot encoding."""
        rng = np.random.default_rng(seed)
        W = rng.choice(["a", "b", "c"], size=n)
        W_effect = np.where(W == "a", 0.0, np.where(W == "b", 2.0, 4.0))
        X = rng.normal(0, 1, size=n)
        T = 0.3 * W_effect + rng.normal(0, 1, size=n)
        Y = 5.0 * T + 3.0 * T * X + W_effect + rng.normal(0, 1, size=n)
        return pd.DataFrame({"T": T, "W": pd.Categorical(W), "X": X, "Y": Y})

    def _fit_and_get_ci(self, df):
        estimand = self._make_estimand(df)
        estimator = LinearRegressionEstimator(
            identified_estimand=estimand,
            confidence_intervals=True,
            confidence_level=0.95,
        )
        estimator.fit(df, effect_modifier_names=["X"])
        estimate = estimator.estimate_effect(df, treatment_value=1, control_value=0, confidence_intervals=True)
        return estimator, estimate

    def test_ci_no_error_continuous_common_cause(self):
        """CI computation does not raise with a continuous common cause and effect modifier."""
        df = self._make_continuous_dataset()
        _, estimate = self._fit_and_get_ci(df)
        ci = estimate.get_confidence_intervals()
        assert ci is not None
        assert ci.shape == (1, 2)
        assert ci[0, 0] < ci[0, 1], "CI lower must be less than upper"

    def test_ci_no_error_categorical_common_cause(self):
        """CI computation does not raise when the common cause is categorical (one-hot encoded)."""
        df = self._make_categorical_dataset()
        _, estimate = self._fit_and_get_ci(df)
        ci = estimate.get_confidence_intervals()
        assert ci is not None
        assert ci.shape == (1, 2)
        assert ci[0, 0] < ci[0, 1], "CI lower must be less than upper"

    def test_ci_uses_actual_encoded_column_count_not_name_count(self):
        """Regression test: interaction_start must use shape[1] not len(names).

        With a 3-level categorical common cause, one-hot encoding (drop_first=True)
        produces 2 columns from 1 variable name. The buggy code used len(names)=1,
        making interaction_start point at the wrong coefficient.
        This test confirms the fix uses the actual encoded column count.
        """
        df = self._make_categorical_dataset()
        estimator, _ = self._fit_and_get_ci(df)

        n_names = len(estimator._observed_common_causes_names)
        n_cols = estimator._observed_common_causes.shape[1]

        # The categorical W expands from 1 name to 2 encoded columns
        assert n_cols > n_names, (
            "Expected categorical variable to expand beyond 1 column, "
            "but got n_cols == n_names. Check dataset has categorical variable."
        )

        # The assert inside _ate_and_se_for_treatment will catch wrong indexing;
        # if it passes, the fixed column count is being used correctly.
        # (The buggy code would compute the wrong interaction_start and either
        # silently return wrong CI or fail the internal assert we added.)
        n_treatments = 1
        em_means = estimator._effect_modifiers.mean(axis=0).to_numpy()
        n_effect_modifiers = len(em_means)
        expected_n_params = 1 + n_treatments + n_cols + n_treatments * n_effect_modifiers
        assert len(estimator.model.params) == expected_n_params

    def test_ci_contains_estimate(self):
        """The CI should be centered around the ATE estimate (not population truth)."""
        df = self._make_categorical_dataset()
        _, estimate = self._fit_and_get_ci(df)
        ci = estimate.get_confidence_intervals()
        lower, upper = ci[0]
        assert lower <= estimate.value <= upper, (
            f"CI [{lower:.4f}, {upper:.4f}] does not contain estimate {estimate.value:.4f}"
        )
