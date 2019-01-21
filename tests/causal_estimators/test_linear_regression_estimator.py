import pytest

from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand
from dowhy.datasets import linear_dataset


class TestLinearRegressionEstimator(object):

    @pytest.mark.parametrize("error_tolerance", [0.01, 0.05])
    def test_average_treatment_effect(self, error_tolerance):
        data = linear_dataset(beta=10,
                              num_common_causes=1,
                              num_instruments=1,
                              num_samples=5000,
                              treatment_is_binary=True)
        true_ate = data["ate"]
        target_estimand = IdentifiedEstimand(
            treatment_variable=data["treatment_name"],
            outcome_variable=data["outcome_name"],
            backdoor_variables=data["common_causes_names"]
        )
        estimator_ate = LinearRegressionEstimator(
            data['df'],
            identified_estimand=target_estimand,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            test_significance=False
        )

        est_ate = estimator_ate.estimate_effect()
        error = est_ate.value - true_ate
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, error_tolerance * 100, est_ate.value, true_ate)
        )
        res = True if (error < true_ate * error_tolerance) else False
        assert res
