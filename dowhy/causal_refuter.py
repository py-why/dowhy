import logging
import numpy as np
import scipy.stats as st

class CausalRefuter:
    
    """Base class for different refutation methods. 

    Subclasses implement specific refutations methods. 

    """

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

    @staticmethod
    def get_estimator_object(new_data, identified_estimand, estimate):
        estimator_class = estimate.params['estimator_class']
        new_estimator = estimator_class(
                new_data,
                identified_estimand,
                identified_estimand.treatment_variable, identified_estimand.outcome_variable, #names of treatment and outcome
                test_significance=None,
                evaluate_effect_strength=False,
                confidence_intervals = estimate.params["confidence_intervals"],
                target_units = estimate.params["target_units"],
                effect_modifiers = estimate.params["effect_modifiers"],
                params = estimate.params["method_params"]
                )
        return new_estimator

    def test_significance(self, estimate, simulations, test_type='auto',significance_level=0.05):
        """
        Tests the satistical significance of the estimate obtained to the simulations produced by a refuter

        The basis behind using the sample statistics of the refuter when we are in fact testing the estimate,
        is due to the fact that, we would ideally expect them to follow the same distribition

        Null Hypothesis: The estimate is a part of the distribution
        Alternative Hypothesis: The estimate does not fall in the distribution.

        Parameters
        -----------
        'estimate': CausalEstimate
        The estimate obtained from the estimator for the original data.
        'simulations': np.array
        An array containing the result of the refuter for the simulations
        'test_type': string, default 'auto'
        The type of test the user wishes to perform.
        'significance_level': float, default 0.05
        The significance level for the statistical test

        Returns
        --------
        significance_dict: Dict
        A Dict containing the p_value and a boolean that indicates if the result is statistically significant
        """
        # Initializing the p_value
        p_value = 0

        if test_type == 'auto':
            num_simulations = len(simulations)
            if num_simulations >= 100: # Bootstrapping
                self.logger.info("Making use of Bootstrap as we have more than 200 examples.\n \
                Note: The greater the number of examples, the more accurate are the confidence estimates")

                # Perform Bootstrap Significance Test with the original estimate and the set of refutations
                p_value = self.perform_bootstrap_test(estimate, simulations)

            else:
                self.logger.warn("We make use of the Normal Distribution as the sample has less than 200 examples.\n \
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
            raise Exception("{} has not been implemented".format(test_type))

        significance_dict = {
                "p_value":p_value,
                "pass_test": p_value <= significance_level
                }

        return significance_dict

    def perform_bootstrap_test(self, estimate, simulations):

        # Get the number of simulations
        num_simulations = len(simulations)
        # Sort the simulations
        simulations.sort()
        # Obtain the median value
        median_refute_values= simulations[int(num_simulations)/2]

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
        self.estimated_effect = estimated_effect,
        self.new_effect = new_effect,
        self.refutation_type = refutation_type

        self.p_value = None

    def __str__(self):
        if self.p_value is None:
            return "{0}\nEstimated effect:{1}\nNew effect:{2}\n".format(
                self.refutation_type, self.estimated_effect, self.new_effect
            )
        else:
            return "{0}\nEstimated effect:{1}\nNew effect:{2}\np value{3}\n".format(
                self.refutation_type, self.estimated_effect, self.new_effect, self.p_value
            )
    
    def add_significance_test_results(self, p_value):
        self.p_value = p_value
