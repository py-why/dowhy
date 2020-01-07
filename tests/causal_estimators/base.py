import itertools

import dowhy.datasets

from dowhy import CausalModel


class TestEstimator(object):
    def __init__(self, error_tolerance, Estimator):
        print("Error tolerance is", error_tolerance)
        self._error_tolerance = error_tolerance
        self._Estimator = Estimator

    def average_treatment_effect_test(self, dataset="linear", beta=10,
            num_common_causes=1, num_instruments=1,
            num_effect_modifiers=0, num_treatments=1,
            num_samples=100000,
            treatment_is_binary=True,
            outcome_is_binary=False,
            method_params=None):
        if dataset == "linear":
            data = dowhy.datasets.linear_dataset(beta=beta,
                                             num_common_causes=num_common_causes,
                                             num_instruments=num_instruments,
                                             num_effect_modifiers = num_effect_modifiers,
                                             num_treatments = num_treatments,
                                             num_samples=num_samples,
                                             treatment_is_binary=treatment_is_binary,
                                             outcome_is_binary = outcome_is_binary)
        elif dataset == "simple-iv":
            data = dowhy.datasets.simple_iv_dataset(beta=beta,
                    num_treatments = num_treatments,
                    num_samples = num_samples,
                    treatment_is_binary=treatment_is_binary,
                    outcome_is_binary = outcome_is_binary)
        else:
            raise ValueError("Dataset type not supported.")

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
            control_value = 0,
            treatment_value = 1,
            test_significance=None,
            evaluate_effect_strength=False,
            confidence_intervals = False,
            target_units = "ate",
            effect_modifiers = data["effect_modifier_names"],
            params=method_params
        )
        true_ate = data["ate"]
        ate_estimate = estimator_ate.estimate_effect()
        error = abs(ate_estimate.value - true_ate)
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, self._error_tolerance * 100, ate_estimate.value, true_ate)
        )
        res = True if (error < abs(true_ate) * self._error_tolerance) else False
        assert res

    def average_treatment_effect_testsuite(self, tests_to_run="all",
            num_common_causes=[2,3], num_instruments=[1,],
            num_effect_modifiers=[0,], num_treatments=[1,],
            treatment_is_binary=[True,],
            outcome_is_binary=[False,],
            dataset = "linear",
            method_params=None):
        args_dict = {
                'num_common_causes': num_common_causes,
                'num_instruments': num_instruments,
                'num_effect_modifiers': num_effect_modifiers,
                'num_treatments': num_treatments,
                'treatment_is_binary': treatment_is_binary,
                'outcome_is_binary': outcome_is_binary
                }
        keys, values = zip(*args_dict.items())
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for cfg in configs:
            print("\nConfig:", cfg)
            cfg['dataset'] = dataset
            cfg['method_params']= method_params
            self.average_treatment_effect_test(**cfg)

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


