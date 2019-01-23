import dowhy.datasets

from dowhy.do_why import CausalModel


class TestEstimator(object):
    def __init__(self, error_tolerance, Estimator):
        print(error_tolerance)
        self._error_tolerance = error_tolerance
        self._Estimator = Estimator
        print(self._error_tolerance)

    def average_treatment_effect_test(self):
        data = dowhy.datasets.linear_dataset(beta=10,
                                             num_common_causes=1,
                                             num_instruments=1,
                                             num_samples=10000,
                                             treatment_is_binary=True)

        model = CausalModel(
            data=data['df'],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None
        )
        target_estimand = model.identify_effect()
        estimator_ate = self._Estimator(
            data['df'],
            identified_estimand=target_estimand,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            test_significance=None
        )
        true_ate = data["ate"]
        ate_estimate = estimator_ate.estimate_effect()
        error = ate_estimate.value - true_ate
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, self._error_tolerance * 100, ate_estimate.value, true_ate)
        )
        res = True if (error < true_ate * self._error_tolerance) else False
        assert res

