import logging
import numpy as np
import scipy.stats as st
import random

from dowhy.utils.api import parse_state

class CausalRefuter:
    """Base class for different refutation methods.

    Subclasses implement specific refutations methods.

    """
    # Default value for the number of simulations to be conducted
    DEFAULT_NUM_SIMULATIONS = 100

    def __init__(self, data, identified_estimand, estimate, **kwargs):
        self._data = data
        self._target_estimand = identified_estimand
        self._estimate = estimate
        self._treatment_name = self._target_estimand.treatment_variable
        self._outcome_name = self._target_estimand.outcome_variable
        self._random_seed = None

        if "random_seed" in kwargs:
            self._random_seed = kwargs['random_seed']
            np.random.seed(self._random_seed)

        self.logger = logging.getLogger(__name__)

        # Concatenate the confounders, instruments and effect modifiers
        self._variables_of_interest = []

        try:
            self._variables_of_interest.extend(self._target_estimand.get_backdoor_variables())
            self._variables_of_interest.extend(self._target_estimand.get_instrumental_variables())
        except AttributeError as attr_error:
            self.logger.error(attr_error)
        
        try:
            self._variables_of_interest.extend(self._estimate.params['effect_modifiers'])
        except AttributeError as attr_error:
            self.logger.error(attr_error)

    def choose_variables(self, required_variables):
        '''Provides a way to choose the confounders whose values we wish to modify

        The variables of interest are the backdoor variables, instrumental variables and effect modifiers.

        - If passed True, will return a list of the variables of interest.
        - If passed False, will return an empty list.
        - If passed a list, will interpret it as a list of variable names. Optionally, all names can be prefixed with '-' to indicate 
            that only variables of interest not in the list should be included.
        - If passed an integer, will return a random sample of that size from the variables of interest.
        '''

        if required_variables is False:
            self.logger.info("No variables requested.")
            return []

        elif required_variables is True:
            self.logger.info("All variables required: Running bootstrap adding noise to confounders, instrumental variables and effect modifiers.")
            return self._variables_of_interest

        elif type(required_variables) is int:
            if len(self._variables_of_interest) < required_variables:
                self.logger.error(f"Too many variables passed. There are {len(self._variables_of_interest)} variable, but {required_variables} was passed.")
                raise ValueError("The number of variables in the required_variables is greater than the number of confounders, instrumental variables and effect modifiers")
            else:
                return random.choices(self._variables_of_interest, k=required_variables)

        elif type(required_variables) is list:

           # Check if all are select or deselect variables
            if all(variable[0] == '-' for variable in required_variables):
                invert = True
                required_variables = [variable[1:] for variable in required_variables]
            elif all(variable[0] != '-' for variable in required_variables):
                invert = False
            else:
                self.logger.error("{} has both select and delect variables".format(required_variables))
                raise ValueError("It appears that there are some select and deselect variables. Note you can either select or delect variables at a time, but not both")

            # Check if all the required_variables belong to confounders, instrumental variables or effect
            invalid_but_required = set(required_variables) - set(self._variables_of_interest)
            if invalid_but_required:
                self.logger.error(f"{invalid_but_required} are not confounder, instrumental variable or effect modifier")
                raise ValueError("At least one of required_variables is not a valid variable name, or it is not a confounder, instrumental variable or effect modifier")

            if invert is False:
                return required_variables
            elif invert is True:
                return list(set(self._variables_of_interest) - set(required_variables))
        
        elif type(required_variables) == str:
            msg = "required_variable cannot be a string. If you want to pass a single variable, pass a one-element list."
            self.logger.error(msg)
            raise TypeError(msg)

        else:
            try:
                return self.choose_variables(list(required_variables))
            except TypeError:
                self.logger.error(f"Incorrect type: {type(required_variables)}. Expected an int, bool or iterable")
                raise TypeError("Expected int, bool or iterable. Got an unexpected datatype")

    def test_significance(self, estimate, simulations, test_type='auto', significance_level=0.05):
        """Determines whether the estimate is statistically significant (two-sided), given the simulated estimates

        This function determines whether `estimate` (a number) is a typical member of `simulations` (a list or numpy array of
        numbers). The idea it to test whether `estimate` could have been generated by the same process as `simulations`.

        The basis behind using the sample statistics of the refuter when we are in fact testing the estimate,
        is due to the fact that, we would ideally expect them to follow the same distribition.

        For refutation tests (e.g., placebo refuters), consider the null distribution as a distribution of effect
        estimates over multiple simulations with placebo treatment, and compute how likely the true estimate (e.g.,
        zero for placebo test) is under the null. If the probability of true effect estimate is lower than the
        p-value, then estimator method fails the test.

        For sensitivity analysis tests (e.g., bootstrap, subset or common cause refuters), the null distribution captures
        the distribution of effect estimates under the "true" dataset (e.g., with an additional confounder or different
        sampling), and we compute the probability of the obtained estimate under this distribution. If the probability is
        lower than the p-value, then the estimator method fails the test.

        Null Hypothesis- The estimate is a part of the distribution
        Alternative Hypothesis- The estimate does not fall in the distribution.

        :param 'estimate': CausalEstimate
            The estimate obtained from the estimator for the original data.
        :param 'simulations': np.array
            An array containing simulated estimates, the result of a refutation
        :param 'test_type': string, default 'auto'
            The type of test the user wishes to perform, one of 'auto', 'bootstrap', 'normal_test'
        :param 'significance_level': float, default 0.05
            The significance level for the statistical test

        :returns: significance_dict: Dict
            A dict containing the `p_value` and a boolean indicating if the result `is_statistically_significant`
        """
        # if auto, determine which type of test to run
        if (test_type == 'auto') and (len(simulations) >= 100):
            test_type = 'bootstrap'
            self.logger.info("Performing bootstrap test as >=100 simulations were passed")
        elif (test_type == 'auto'):
            test_type = 'normal_test'
            self.logger.info("Performing normal test as <100 simulations were passed")

        # run the test
        if test_type == 'bootstrap':
            p_value = self.perform_bootstrap_test(estimate, simulations)
        elif test_type == 'normal_test':
            p_value = self.perform_normal_distribution_test(estimate, simulations)
        else:
            raise NotImplementedError(f"test_type should be one of 'auto', 'bootstrap' or 'normal_test'; instead got {test_type}")

        significance_dict = {
                "p_value":p_value,
                "is_statistically_significant": p_value <= significance_level
                }
        return significance_dict

    def perform_bootstrap_test(self, estimate, simulations):

        num_simulations = len(simulations)
        simulations.sort()
        median_refute_values = simulations[int(num_simulations/2)]

        # Performing a two sided test
        if estimate.value > median_refute_values:
            # np.searchsorted tells us the index if it were a part of the array
            # We select side to be left as we want to find the first value that matches
            estimate_index = np.searchsorted(simulations, estimate.value, side="left")
            # We subtact 1 as we are finding the value from the right tail
            quantile = 1 - (estimate_index/num_simulations)
        else:
            # We take the side to be right as we want to find the last index that matches
            estimate_index = np.searchsorted(simulations, estimate.value, side="right")
            # We get the probability with respect to the left tail.
            quantile = estimate_index/num_simulations

        # two-tailed test, so we need to multiply by 2
        p_value = quantile * 2
        return p_value

    def perform_normal_distribution_test(self, estimate, simulations):

        print("estimate", estimate.value)
        print("sims", simulations)

        mean_refute_values = np.mean(simulations)
        std_dev_refute_values = np.std(simulations)
        z_score = (estimate.value - mean_refute_values)/ std_dev_refute_values

        print(mean_refute_values, std_dev_refute_values, z_score)

        if z_score > 0: # Right Tail
            quantile = 1 - st.norm.cdf(z_score)
        else: # Left Tail
            quantile = st.norm.cdf(z_score)

        # two-tailed test, so we need to multiply by 2
        p_value = quantile * 2
        return p_value

    def refute_estimate(self):
        raise NotImplementedError


class CausalRefutation:
    """Class for storing the result of a refutation method
    
    :ivar estimated_effect float:
        The estimated effect of the refutation
    :ivar new_effect float:
        The new effect from the refutation
    :ivar refutation_type string:
        description of the type of refutation
    :refutation_result:
        (not used)
    """
    def __init__(self, estimated_effect, new_effect, refutation_type):
        self.estimated_effect = estimated_effect
        self.new_effect = new_effect
        self.refutation_type = refutation_type
        self.refutation_result = None

    def add_significance_test_results(self, refutation_result):
        self.refutation_result = refutation_result

    def add_refuter(self, refuter_instance):
        self.refuter = refuter_instance

    def interpret(self, method_name=None, **kwargs):
        """Interpret the refutation results.

        :param method_name: Method used (string) or a list of methods. If None, then the default for the specific refuter is used.
        :returns: None
        """
        if method_name is None:
            method_name = self.refuter.interpret_method
        method_name_arr = parse_state(method_name)
        import dowhy.interpreters as interpreters
        for method in method_name_arr:
            interpreter = interpreters.get_class_object(method)
            interpreter(self, **kwargs).interpret()

    def __str__(self):
        if self.refutation_result is None:
            return f"{self.refutation_type}\nEstimated effect:{self.estimated_effect}\nNew effect:{self.new_effect}\n"
        else:
            return (f"{self.refutation_type}\nEstimated effect:{self.estimated_effect}\n"
            f"New effect:{self.new_effect}\np value:{self.refutation_result['p_value']}\n")
