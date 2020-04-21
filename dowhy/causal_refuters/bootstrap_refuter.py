from dowhy.causal_refuter import CausalRefuter, CausalRefutation
from dowhy.causal_estimator import CausalEstimator
import numpy as np
import random
from sklearn.utils import resample
import logging

class BootstrapRefuter(CausalRefuter):
    """
    Refute an estimate by running it on a random sample of the data containing measurement error in the 
    confounders. This allows us to find the ability of the estimator to find the effect of the 
    treatment on the outcome.
    
    It supports additional parameters that can be specified in the refute_estimate() method.
    
    Parameters
    -----------
    -'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'sample_size': int, Size of the original data by default
    The size of each bootstrap sample
    - 'required_variables': int, list, bool, True by default
    A user can input either an integer value, list or bool.
        1. An integer argument refers to how many variables will be modified
        2. A list allows the user to explicitly refer to which variables should be selected to be made noisy
            Furthermore, a user can either choose to select the variables desired. Or they can deselect the variables,
            that they do not want in their analysis. 
            For example:
            We need to pass required_variables = [W0,W1] is we want W0 and W1.
            We need to pass required_variables = [-W0,-W1] if we want all variables excluding W0 and W1.
        3. If the user passes True, noise is added to  confounders, instrumental variables and effect modifiers
           If the value is False, we just Bootstrap the existing dataset  
    - 'noise': float, BootstrapRefuter.DEFAULT_STD_DEV by default
    The standard deviation of the noise to be added to the data
    - 'probability_of_change': float, 'noise' by default if the value is less than 1
    It specifies the probability with which we change the data for a boolean or categorical variable
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
        required_variables = kwargs.pop("required_variables", True)
        self._noise = kwargs.pop("noise", BootstrapRefuter.DEFAULT_STD_DEV )
        self._probability_of_change = kwargs.pop("probability_of_change", None)
        self._random_state = kwargs.pop("random_state", None)

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._chosen_variables = self.choose_variables(required_variables)

        if self._chosen_variables is None:
            self.logger.info("INFO: There are no chosen variables")
        else:    
            self.logger.info("INFO: The chosen variables are: " +
                            ",".join(self._chosen_variables))

        if self._probability_of_change is None:
            if self._noise > 1:
                self.logger.error("Error in using noise:{} for Binary Flip. The value is greater than 1".format(self._noise))
                raise ValueError("The value for Binary Flip cannot be greater than 1")
            else:
                self._probability_of_change = self._noise
        elif self._probability_of_change > 1:
            self.logger.error("The probability of flip is: {}, However, this value cannot be greater than 1".format(self._probability_of_change))
            raise ValueError("Probability of Flip cannot be greater than 1")

    
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
                        scaling_factor = new_data[variable].std() 
                        new_data[variable] += np.random.normal(loc=0.0, scale=self._noise * scaling_factor,size=self._sample_size) 
                    
                    elif 'bool' in new_data[variable].dtype.name:
                        probs = np.random.uniform(0, 1, self._sample_size )
                        new_data[variable] = np.where(probs < self._probability_of_change, 
                                                        np.logical_not(new_data[variable]), 
                                                        new_data[variable]) 
                    
                    elif 'category' in new_data[variable].dtype.name:
                        categories = new_data[variable].unique()
                        # Find the set difference for each row
                        changed_data = new_data[variable].apply( lambda row: list( set(categories) - set([row]) ) )
                        # Choose one out of the remaining
                        changed_data = changed_data.apply( lambda row: random.choice(row)  )
                        new_data[variable] = np.where(probs < self._probability_of_change, changed_data)
                        new_data[variable].astype('category')

            new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            sample_estimates[index] = new_effect.value

        refute = CausalRefutation(
            self._estimate.value,
            np.mean(sample_estimates),
            refutation_type="Refute: Bootstrap Sample Dataset"
        )

        # We want to see if the estimate falls in the same distribution as the one generated by the refuter
        # Ideally that should be the case as running bootstrap should not have a significant effect on the ability
        # of the treatment to affect the outcome
        refute.add_significance_test_results(
            self.test_significance(self._estimate, sample_estimates)
        )

        return refute

    