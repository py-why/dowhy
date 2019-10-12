import dowhy.datasets

from dowhy import CausalModel


class TestEstimator(object):
    def __init__(self, error_tolerance, Estimator):
        print(error_tolerance)
        self._error_tolerance = error_tolerance
        self._Estimator = Estimator
        print(self._error_tolerance)

    def average_treatment_effect_test_binary(self, dataset="linear", beta=10,
            num_common_causes=1, num_instruments=1, num_samples=100000,
            treatment_is_binary=True):
        data = dowhy.datasets.linear_dataset(beta=beta,
                                             num_common_causes=num_common_causes,
                                             num_instruments=num_instruments,
                                             num_samples=num_samples,
                                             treatment_is_binary=treatment_is_binary)

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
        error = abs(ate_estimate.value - true_ate)
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, self._error_tolerance * 100, ate_estimate.value, true_ate)
        )
        res = True if (error < abs(true_ate) * self._error_tolerance) else False
        assert res

    def average_treatment_effect_test_continuous(self, dataset="linear", beta=1,
            num_common_causes=3, num_instruments=2, num_samples=100000,
            treatment_is_binary=False):
        data = dowhy.datasets.linear_dataset(beta=beta,
                                             num_common_causes=num_common_causes,
                                             num_instruments=num_instruments,
                                             num_samples=num_samples,
                                             treatment_is_binary=treatment_is_binary)
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
        error = abs(ate_estimate.value - true_ate)
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, self._error_tolerance * 100, ate_estimate.value, true_ate)
        )
        res = True if (error < abs(true_ate) * self._error_tolerance) else False
        assert res


    def average_treatment_effect_testsuite(self, tests_to_run="all"):
        self.average_treatment_effect_test_binary(num_common_causes=1)
        if tests_to_run != "atleast-one-common-cause":
            self.average_treatment_effect_test_binary(num_common_causes=0)

    def custom_data_average_treatment_effect_test(self, data):
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


