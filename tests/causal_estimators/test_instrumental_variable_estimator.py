import pytest
import dowhy.datasets

from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
from dowhy.do_why import CausalModel


class TestInstrumentalVariableEstimator(object):

    @pytest.mark.parametrize("error_tolerance", [0.05, 0.1])
    def test_average_treatment_effect(self, error_tolerance):
        data = dowhy.datasets.linear_dataset(beta=10,
                                             num_common_causes=1,
                                             num_instruments=1,
                                             num_samples=5000,
                                             treatment_is_binary=True)

        model = CausalModel(
            data=data['df'],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True
        )
        true_ate = data["ate"]
        target_estimand = model.identify_effect()
        estimator_ate = InstrumentalVariableEstimator(
            data['df'],
            identified_estimand=target_estimand,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            test_significance=False
        )

        ate_estimate = estimator_ate.estimate_effect()
        error = ate_estimate.value - true_ate
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, error_tolerance * 100, ate_estimate.value, true_ate)
        )
        res = True if (error < true_ate * error_tolerance) else False
        assert res
