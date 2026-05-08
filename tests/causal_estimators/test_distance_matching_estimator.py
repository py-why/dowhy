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
            confidence_intervals=[False],
            test_significance=[False],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )


def test_distance_metric_params_mahalanobis():
    """Regression test: V and VI must be routed through metric_params, not top-level kwargs.

    NearestNeighbors does not accept V/VI/w as top-level constructor arguments;
    they must be passed via metric_params=. This test catches the regression where
    **self.distance_metric_params was splatted directly into NearestNeighbors.__init__,
    causing a TypeError for any metric that requires extra parameters (e.g. Mahalanobis).
    """
    import dowhy.datasets
    from dowhy import EstimandType, identify_effect_auto
    from dowhy.graph import build_graph_from_str

    data = dowhy.datasets.linear_dataset(
        beta=5,
        num_common_causes=2,
        num_instruments=0,
        num_effect_modifiers=0,
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

    # 2 common causes → 2x2 covariance matrix for Mahalanobis
    V = np.eye(2)
    estimator = DistanceMatchingEstimator(
        identified_estimand=target_estimand,
        distance_metric="mahalanobis",
        V=V,
    )
    estimator.fit(data["df"])
    # Should not raise TypeError about unexpected keyword argument 'V'
    result = estimator.estimate_effect(data["df"], treatment_value=1, control_value=0, target_units="att")
    assert result.value is not None
