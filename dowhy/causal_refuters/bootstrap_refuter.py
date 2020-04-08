from dowhy.causal_refuter import CausalRefuter, CausalRefutation
import numpy as np
import random
from sklearn.utils import resample
import logging

class BootstrapRefuter(CausalRefuter):
    """
    Refute an estimate by running it on a random sample of the original data.
    It supports additional parameters that can be specified in the refute_estimate() method.
    -'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'sample_size': int, Size of the original data by default
    The size of each bootstrap sample
    - 'required_variables': int, list
    An integer argument means that we select a select number of covariates out of all covariates
    - 'noise': float, BootstrapRefuter.DEFAULT_STD_DEV by default
    The standard deviation of the noise to be added to the data
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. For this purpose, 
    we repeat the same seed in the psuedo-random generator.
    """

    DEFAULT_STD_DEV = 0.1
    DEFAULT_SUCCESS_PROBABILITY = 0.5
    DEFAULT_NUMBER_OF_TRIALS = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS )
        self._sample_size = kwargs.pop("sample_size", len(self._data))
        self._required_variables = kwargs.pop("required_variables", None)
        self._noise = kwargs.pop("noise", BootstrapRefuter.DEFAULT_STD_DEV )
        self._random_state = kwargs.pop("random_state", None)

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sanity check the data passed by the user
        # If the data is invalid, we run the default behavior
        if self._required_variables is int:
            if len(self._target_estimand.backdoor_variables) < self._required_variables:
                self.logger.warning("Too many variables passed.\n The number of backdoor variables is: {}.\n The number of variables passed: {}".format(
                    len(self._target_estimand.backdoor_variables),
                    self._required_variables )
                )
                self.logger.warning("The bootstrap refuter will follow the default behavior")
                self._required_variables = None

        elif self._required_variables is list:
            for variable in self._required_variables:
                self.logger.warning(variable in self._target_estimand.backdoor_variables), "The variable {} is not not a backdoor variable".format(variable)
                break
            self.logger.warning("The bootstrap refuter will follow the default behavior")
            self._required_variables = None
        
        else:
            self.logger.warning("Incorrect type: {}. Expected an int or list".format( type(self._required_variables) ) )
            self._required_variables = None

        
        self.choose_desired_variables()

    def refute_estimate(self, *args, **kwargs):
        if self._sample_size > len(self._data):
                self.logger.warning("The sample size is larger than the population size")

        sample_estimates = np.zeros(self._num_simulations)
        self.logger.info("Refutation over {} simulated datasets of size {} each"
                         .format(self._num_simulations
                         ,self._sample_size )
                        ) 
        
        for index in range(self._num_simulations):
            if self._random_state is None:
                new_data = resample(self._data, 
                                n_samples=self._sample_size )
            else:
                new_data = resample(self._data,
                                    n_samples=self._sample_size,
                                    random_state=self._random_state )

            if self._chosen_variables is not None:
                for variable in self._chosen_variables:
                    
                    if ('float' or 'int') in new_data[variable].dtype.name: 
                        new_data[variable] += np.random.randn(self._sample_size) * self._noise
                    
                    elif 'bool' in new_data[variable]:
                        change_mask = self.get_mask_variables()
                        # Set these values to zero
                        new_data[variable] *= change_mask
                        # Sample a binomial distribution and set the value for those datapoints we set to zero in the previous step
                        new_data[variable] += (1 - change_mask) * np.random.binomial(BootstrapRefuter.DEFAULT_NUMBER_OF_TRIALS,
                                                                  BootstrapRefuter.DEFAULT_SUCCESS_PROBABILITY,
                                                                  self._sample_size).astype(bool) 
                    
                    elif 'category' in new_data[variable]:
                        change_mask = self.get_mask_variables()
                        # Set these values to zero
                        new_data[variable] *= change_mask
                        # Sample a binomial distribution and set the value for those datapoints we set to zero in the previous step
                        new_data[variable] += (1 - change_mask) * np.random.choice(categories, size=self._sample_size)
                        new_data[variable].astype('category')

            new_estimator = self.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            sample_estimates[index] = new_effect.value

        refute = CausalRefutation(
            self._estimate.value,
            np.mean(sample_estimates),
            refutation_type="Refute: Bootstrap Sample Dataset"
        )

        # We want to see if the estimate falls in the same distribution as the one generated by the refuter
        # Ideally that should be the case as bootstrapping should not have a significant effect on the ability
        # of the treatment to affect the outcome
        refute.add_significance_test_results(
            self.test_significance(self._estimate, sample_estimates)
        )

        return refute

    def choose_desired_variables(self):
        '''
            This method provides a way to choose the confounders whose values we wish to
            modify for finding its effect on the ability of the treatment to affect the outcome.
        '''
        if self._required_variables is None:
            self._chosen_variables = None
        elif type(self._required_variables) is int:
            # Shuffle the confounders 
            random.shuffle(self._target_estimand.backdoor_variables)
            self._chosen_variables = self._target_estimand.backdoor_variables[:self._required_variables]
        elif type(self._required_variables) is list:
            self._chosen_variables = self._required_variables

    def get_mask_variables(self):
        '''
            This function helps to create a mask that sets false for all values that have a value
            smaller than the noise level set by the user.
        '''
        # Sample a uniform distribution for each data point
        probs = np.random.uniform(0,1, self._sample_size)
        # Find all the data points in which the value is smaller than equal to the value set by the user
        change_mask = probs <= self._noise
        return change_mask