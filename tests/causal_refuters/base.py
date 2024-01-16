import logging

import dowhy.datasets
from dowhy import CausalModel


class SimpleRefuter(object):
    def __init__(
        self,
        error_tolerance,
        estimator_method,
        refuter_method,
        transformations=None,
        params=None,
        confounders_effect_on_t=None,
        confounders_effect_on_y=None,
        effect_strength_on_t=None,
        effect_strength_on_y=None,
        **kwargs,
    ):
        self._error_tolerance = error_tolerance
        self.estimator_method = estimator_method
        self.identifier_method = estimator_method.split(".")[0]
        self.refuter_method = refuter_method
        self.transformations = transformations
        self.params = params
        self.confounders_effect_on_t = confounders_effect_on_t
        self.confounders_effect_on_y = confounders_effect_on_y
        self.effect_strength_on_t = effect_strength_on_t
        self.effect_strength_on_y = effect_strength_on_y
        self.logger = logging.getLogger(__name__)
        self.logger.debug(self._error_tolerance)

    def null_refutation_test(
        self,
        data=None,
        dataset="linear",
        beta=10,
        num_common_causes=1,
        num_instruments=1,
        num_samples=100000,
        treatment_is_binary=True,
        treatment_is_category=False,
        num_dummyoutcome_simulations=None,
    ):
        # Supports user-provided dataset object
        if data is None:
            data = dowhy.datasets.linear_dataset(
                beta=beta,
                num_common_causes=num_common_causes,
                num_instruments=num_instruments,
                num_samples=num_samples,
                treatment_is_binary=treatment_is_binary,
                treatment_is_category=treatment_is_category,
            )

        print(data["df"])

        print("")
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        target_estimand = model.identify_effect(method_name="exhaustive-search")
        target_estimand.set_identifier_method(self.identifier_method)
        ate_estimate = model.estimate_effect(
            identified_estimand=target_estimand, method_name=self.estimator_method, test_significance=None
        )
        true_ate = data["ate"]
        self.logger.debug(true_ate)

        if self.refuter_method == "add_unobserved_common_cause":
            # To test if there are any exceptions
            ref = model.refute_estimate(
                target_estimand,
                ate_estimate,
                method_name=self.refuter_method,
                confounders_effect_on_treatment=self.confounders_effect_on_t,
                confounders_effect_on_outcome=self.confounders_effect_on_y,
                effect_strength_on_treatment=self.effect_strength_on_t,
                effect_strength_on_outcome=self.effect_strength_on_y,
            )
            self.logger.debug(ref.new_effect)

            # To test if the estimate is identical if refutation parameters are zero
            refute = model.refute_estimate(
                target_estimand,
                ate_estimate,
                method_name=self.refuter_method,
                confounders_effect_on_treatment=self.confounders_effect_on_t,
                confounders_effect_on_outcome=self.confounders_effect_on_y,
                effect_strength_on_treatment=0,
                effect_strength_on_outcome=0,
            )
            error = abs(refute.new_effect - ate_estimate.value)

            print(
                "Error in refuted estimate = {0} with tolerance {1}%. Estimated={2},After Refutation={3}".format(
                    error, self._error_tolerance * 100, ate_estimate.value, refute.new_effect
                )
            )
            res = True if (error < abs(ate_estimate.value) * self._error_tolerance) else False
            assert res

        elif self.refuter_method == "placebo_treatment_refuter":
            if treatment_is_binary is True:
                ref = model.refute_estimate(
                    target_estimand, ate_estimate, method_name=self.refuter_method, num_simulations=10
                )
            else:
                ref = model.refute_estimate(target_estimand, ate_estimate, method_name=self.refuter_method)
            # This value is hardcoded to be zero as we are runnning this on a linear dataset.
            # Ordinarily, we should expect this value to be zero.
            EXPECTED_PLACEBO_VALUE = 0

            error = abs(ref.new_effect - EXPECTED_PLACEBO_VALUE)

            print(
                "Error in the refuted estimate = {0} with tolerence {1}. Expected Value={2}, After Refutation={3}".format(
                    error, self._error_tolerance, EXPECTED_PLACEBO_VALUE, ref.new_effect
                )
            )

            print(ref)

            res = True if (error < self._error_tolerance) else False
            assert res

        elif self.refuter_method == "data_subset_refuter":
            if treatment_is_binary is True:
                ref = model.refute_estimate(
                    target_estimand, ate_estimate, method_name=self.refuter_method, num_simulations=5
                )
            else:
                ref = model.refute_estimate(target_estimand, ate_estimate, method_name=self.refuter_method)

            error = abs(ref.new_effect - ate_estimate.value)

            print(
                "Error in the refuted estimate = {0} with tolerence {1}%. Estimated={2}, After Refutation={3}".format(
                    error, self._error_tolerance * 100, ate_estimate.value, ref.new_effect
                )
            )

            print(ref)

            res = True if (error < abs(ate_estimate.value) * self._error_tolerance) else False
            assert res

        elif self.refuter_method == "bootstrap_refuter":
            if treatment_is_binary is True:
                ref = model.refute_estimate(
                    target_estimand, ate_estimate, method_name=self.refuter_method, num_simulations=5
                )
            else:
                ref = model.refute_estimate(target_estimand, ate_estimate, method_name=self.refuter_method)

            error = abs(ref.new_effect - ate_estimate.value)

            print(
                "Error in the refuted estimate = {0} with tolerence {1}%. Estimated={2}, After Refutation={3}".format(
                    error, self._error_tolerance * 100, ate_estimate.value, ref.new_effect
                )
            )

            print(ref)

            res = True if (error < abs(ate_estimate.value) * self._error_tolerance) else False
            assert res

        elif self.refuter_method == "dummy_outcome_refuter":
            if self.transformations is None:
                ref_list = model.refute_estimate(
                    target_estimand,
                    ate_estimate,
                    method_name=self.refuter_method,
                    num_simulations=num_dummyoutcome_simulations,
                )
            else:
                ref_list = model.refute_estimate(
                    target_estimand,
                    ate_estimate,
                    method_name=self.refuter_method,
                    transformation_list=self.transformations,
                    params=self.params,
                    num_simulations=num_dummyoutcome_simulations,
                )

            INDEX = 0
            ref = ref_list[INDEX]

            # This value is hardcoded to be zero as we are runnning this on a linear dataset.
            # Ordinarily, we should expect this value to be zero.
            EXPECTED_DUMMY_OUTCOME_VALUE = 0

            error = abs(ref.new_effect - EXPECTED_DUMMY_OUTCOME_VALUE)

            print(
                "Error in the refuted estimate = {0} with tolerence {1}. Expected Value={2}, After Refutation={3}".format(
                    error, self._error_tolerance, EXPECTED_DUMMY_OUTCOME_VALUE, ref.new_effect
                )
            )

            print(ref)

            res = True if (error < self._error_tolerance) else False
            assert res

    def binary_treatment_testsuite(
        self, num_samples=100000, num_common_causes=1, tests_to_run="all", num_dummyoutcome_simulations=2
    ):
        self.null_refutation_test(
            num_common_causes=num_common_causes,
            num_samples=num_samples,
            num_dummyoutcome_simulations=num_dummyoutcome_simulations,
        )
        if tests_to_run != "atleast-one-common-cause":
            self.null_refutation_test(
                num_common_causes=0, num_samples=num_samples, num_dummyoutcome_simulations=num_dummyoutcome_simulations
            )

    def continuous_treatment_testsuite(
        self, num_samples=100000, num_common_causes=1, tests_to_run="all", num_dummyoutcome_simulations=2
    ):
        self.null_refutation_test(
            num_common_causes=num_common_causes,
            num_samples=num_samples,
            treatment_is_binary=False,
            num_dummyoutcome_simulations=num_dummyoutcome_simulations,
        )
        if tests_to_run != "atleast-one-common-cause":
            self.null_refutation_test(
                num_common_causes=0,
                num_samples=num_samples,
                treatment_is_binary=False,
                num_dummyoutcome_simulations=num_dummyoutcome_simulations,
            )

    def categorical_treatment_testsuite(
        self, num_samples=100000, num_common_causes=1, tests_to_run="all", num_dummyoutcome_simulations=2
    ):
        self.null_refutation_test(
            num_common_causes=num_common_causes,
            num_samples=num_samples,
            treatment_is_binary=False,
            treatment_is_category=True,
            num_dummyoutcome_simulations=num_dummyoutcome_simulations,
        )
        if tests_to_run != "atleast-one-common-cause":
            self.null_refutation_test(
                num_common_causes=0,
                num_samples=num_samples,
                treatment_is_binary=False,
                treatment_is_category=True,
                num_dummyoutcome_simulations=num_dummyoutcome_simulations,
            )
