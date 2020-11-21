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

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Concatenate the confounders, instruments and effect modifiers
        try:
            self._variables_of_interest = self._target_estimand.get_backdoor_variables() + \
                                        self._target_estimand.instrumental_variables + \
                                        self._estimate.params['effect_modifiers']
        except AttributeError as attr_error:
            self.logger.error(attr_error)

    def choose_variables(self, required_variables):
        '''
            This method provides a way to choose the confounders whose values we wish to
            modify for finding its effect on the ability of the treatment to affect the outcome.
        '''

        invert = None

        if required_variables is False:

            self.logger.info("All variables required: Running bootstrap adding noise to confounders, instrumental variables and effect modifiers.")
            return None

        elif required_variables is True:

            self.logger.info("All variables required: Running bootstrap adding noise to confounders, instrumental variables and effect modifiers.")
            return self._variables_of_interest

        elif type(required_variables) is int:

            if len(self._variables_of_interest) < required_variables:
                self.logger.error("Too many variables passed.\n The number of  variables is: {}.\n The number of variables passed: {}".format(
                    len(self._variables_of_interest),
                    required_variables )
                )
                raise ValueError("The number of variables in the required_variables is greater than the number of confounders, instrumental variables and effect modifiers")
            else:
                # Shuffle the confounders
                random.shuffle(self._variables_of_interest)
                return self._variables_of_interest[:required_variables]

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
            if set(required_variables) - set(self._variables_of_interest) != set([]):
                self.logger.error("{} are not confounder, instrumental variable or effect modifier".format( list( set(required_variables) - set(self._variables_of_interest) ) ))
                raise ValueError("At least one of required_variables is not a valid variable name, or it is not a confounder, instrumental variable or effect modifier")

            if invert is False:
                return required_variables
            elif invert is True:
                return list( set(self._variables_of_interest) - set(required_variables) )

        else:
            self.logger.error("Incorrect type: {}. Expected an int,list or bool".format( type(required_variables) ) )
            raise TypeError("Expected int, list or bool. Got an unexpected datatype")

    def test_significance(self, estimate, simulations, test_type='auto',significance_level=0.05):
        """ Tests the statistical significance of the estimate obtained to the simulations produced by a refuter.

        The basis behind using the sample statistics of the refuter when we are in fact testing the estimate,
        is due to the fact that, we would ideally expect them to follow the same distribition

        For refutation tests (e.g., placebo refuters), consider the null distribution as a distribution of effect
        estimates over multiple simulations with placebo treatment, and compute how likely the true estimate (e.g.,
        zero for placebo test) is under the null. If the probability of true effect estimate is lower than the
        p-value, then estimator method fails the test.

        For sensitivity analysis tests (e.g., bootstrap, subset or common cause refuters), the null distribution captures
        the distribution of effect estimates under the "true" dataset (e.g., with an additional confounder or different
        sampling), and we compute the probability of the obtained estimate under this distribution. If the probability is
        lower than the p-value, then the estimator method fails the test

        Null Hypothesis: The estimate is a part of the distribution
        Alternative Hypothesis: The estimate does not fall in the distribution.

        :param 'estimate': CausalEstimate
        The estimate obtained from the estimator for the original data.
        :param 'simulations': np.array
        An array containing the result of the refuter for the simulations
        :param 'test_type': string, default 'auto'
        The type of test the user wishes to perform.
        :param 'significance_level': float, default 0.05
        The significance level for the statistical test

        :returns: significance_dict: Dict
        A Dict containing the p_value and a boolean that indicates if the result is statistically significant
        """
        # Initializing the p_value
        p_value = 0

        if test_type == 'auto':
            num_simulations = len(simulations)
            if num_simulations >= 100: # Bootstrapping
                self.logger.info("Making use of Bootstrap as we have more than 100 examples.\n \
                Note: The greater the number of examples, the more accurate are the confidence estimates")

                # Perform Bootstrap Significance Test with the original estimate and the set of refutations
                p_value = self.perform_bootstrap_test(estimate, simulations)

            else:
                self.logger.warning("We assume a Normal Distribution as the sample has less than 100 examples.\n \
                Note: The underlying distribution may not be Normal. We assume that it approaches normal with the increase in sample size.")

                # Perform Normal Tests of Significance with the original estimate and the set of refutations
                p_value = self.perform_normal_distribution_test(estimate, simulations)

        elif test_type == 'bootstrap':
            self.logger.info("Performing Bootstrap Test with {} samples\n \
            Note: The greater the number of examples, the more accurate are the confidence estimates".format( len(simulations) ) )

            # Perform Bootstrap Significance Test with the original estimate and the set of refutations
            p_value = self.perform_bootstrap_test(estimate, simulations)

        elif test_type == 'normal_test':
            self.logger.info("Performing Normal Test with {} samples\n \
            Note: We assume that the underlying distribution is Normal.".format( len(simulations) ) )

            # Perform Normal Tests of Significance with the original estimate and the set of refutations
            p_value = self.perform_normal_distribution_test(estimate, simulations)

        else:
            raise NotImplementedError

        significance_dict = {
                "p_value":p_value,
                "is_statistically_significant": p_value <= significance_level
                }

        return significance_dict

    def perform_bootstrap_test(self, estimate, simulations):

        # Get the number of simulations
        num_simulations = len(simulations)
        # Sort the simulations
        simulations.sort()
        # Obtain the median value
        median_refute_values= simulations[int(num_simulations/2)]

        # Performing a two sided test
        if estimate.value > median_refute_values:
            # np.searchsorted tells us the index if it were a part of the array
            # We select side to be left as we want to find the first value that matches
            estimate_index = np.searchsorted(simulations, estimate.value, side="left")
            # We subtact 1 as we are finding the value from the right tail
            p_value = 1 - (estimate_index/ num_simulations)
        else:
            # We take the side to be right as we want to find the last index that matches
            estimate_index = np.searchsorted(simulations, estimate.value, side="right")
            # We get the probability with respect to the left tail.
            p_value = estimate_index / num_simulations

        return p_value

    def perform_normal_distribution_test(self, estimate, simulations):
        # Get the mean for the simulations
        mean_refute_values = np.mean(simulations)
        # Get the standard deviation for the simulations
        std_dev_refute_values = np.std(simulations)
        # Get the Z Score [(val - mean)/ std_dev ]
        z_score = (estimate.value - mean_refute_values)/ std_dev_refute_values


        if z_score > 0: # Right Tail
            p_value = 1 - st.norm.cdf(z_score)
        else: # Left Tail
            p_value = st.norm.cdf(z_score)

        return p_value

    def refute_estimate(self):
        raise NotImplementedError


class CausalRefutation:
    """Class for storing the result of a refutation method.

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
            return "{0}\nEstimated effect:{1}\nNew effect:{2}\n".format(
                self.refutation_type, self.estimated_effect, self.new_effect
            )
        else:
            return "{0}\nEstimated effect:{1}\nNew effect:{2}\np value:{3}\n".format(
                self.refutation_type, self.estimated_effect, self.new_effect, self.refutation_result['p_value']
            )

