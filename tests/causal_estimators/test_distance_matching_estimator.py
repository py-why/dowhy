import numpy as np
import pytest

from dowhy.causal_estimators.distance_matching_estimator import DistanceMatchingEstimator

from .base import SimpleEstimator


@pytest.mark.usefixtures("fixed_seed")
class TestDistanceMatchingEstimator:
    @pytest.mark.parametrize(
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
                0.3,
                DistanceMatchingEstimator,
                [1, 2],
                [0],
                [0],
                [1],
                [True],
                [False],
                "backdoor",
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
        )

    def test_distance_metric_params_passed_to_estimator(self):
        """Regression test for https://github.com/py-why/dowhy/issues/1390.

        distance_metric_params such as V (for Mahalanobis) must be forwarded
        from method_params to NearestNeighbors, not silently dropped.
        """
        import dowhy.datasets
        from dowhy import EstimandType, identify_effect_auto
        from dowhy.graph import build_graph_from_str

        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=2,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        graph = build_graph_from_str(data["gml_graph"])
        observed_nodes = list(data["df"].columns)
        identified_estimand = identify_effect_auto(
            graph,
            data["treatment_name"],
            data["outcome_name"],
            observed_nodes,
            EstimandType.NONPARAMETRIC_ATE,
        )
        common_causes = data["df"][data["common_causes_names"]].values
        V = np.cov(common_causes.T)

        estimator = DistanceMatchingEstimator(
            identified_estimand=identified_estimand,
            distance_metric="mahalanobis",
            V=V,
        )
        assert estimator.distance_metric_params == {"V": V}, (
            "distance_metric_params should capture V from kwargs"
        )

        # Also verify that fit + estimate_effect works end-to-end without error.
        estimator.fit(data["df"])
        estimate = estimator.estimate_effect(data["df"], target_units="att")
        assert estimate.value is not None
