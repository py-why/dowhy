import numpy as np
import pandas as pd
import pytest

from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class TestInstrumentalVariableEstimator(object):

    @pytest.mark.parametrize("error_tolerance", [0.01, 0.05])
    def test_average_treatment_effect(self, linear_dataset, error_tolerance):
        data = linear_dataset["data"]
        true_ate = linear_dataset["ate"]
        target_estimand = IdentifiedEstimand(
            treatment_variable=linear_dataset["treatment"],
            outcome_variable=linear_dataset["outcome"],
            backdoor_variables=linear_dataset["common_causes"]
        )
        estimator_ate = LinearRegressionEstimator(
            data,
            identified_estimand=target_estimand,
            treatment=linear_dataset["treatment"],
            outcome=linear_dataset["outcome"]
        )

        est_ate = estimator_ate.estimate_effect()
        error = est_ate.value - true_ate
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, error_tolerance * 100, est_ate.value, true_ate)
        )
        res = True if (error < true_ate * error_tolerance) else False
        assert res
