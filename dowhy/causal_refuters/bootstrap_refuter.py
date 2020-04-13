from dowhy.causal_refuter import CausalRefuter, CausalRefutation
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
    - 'required_variables': int, list
    A user can input either an integer value or a list.
        1. An integer argument refers to how many confounders  will be modified
        2. A list allows the user to explicitly refer to which confounders should be seleted to be made noisy
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
        self._required_variables = kwargs.pop("required_variables", None)
        self._noise = kwargs.pop("noise", BootstrapRefuter.DEFAULT_STD_DEV )
        self._probability_of_change = kwargs.pop("probability_of_change", None)
        self._random_state = kwargs.pop("random_state", None)
        self._invert = None
        # Concatenate the confounders, instruments and effect modifiers
        self._variables_of_interest = self._target_estimand.backdoor_variables + \
                                      self._target_estimand.instrumental_variables + \
                                      self._estimate.params['effect_modifiers']

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sanity check the parameters passed by the user
        # If the data is invalid, we throw the corresponding error
        if self._required_variables is int:
            if len(self._variables_of_interest) < self._required_variables:
                self.logger.error("Too many variables passed.\n The number of  variables is: {}.\n The number of variables passed: {}".format(
                    len(self._variables_of_interest),
                    self._required_variables )
                )
                raise ValueError("The number of variables in the arguments is greater than the number of variables")

        elif self._required_variables is list:
            for variable in self._required_variables:
                # Find out if the user wants to select or deselect the variable
                if self._invert is None:
                    if variable[0] == '-':
                        self._invert = True
                    else:
                        self._invert = False

                if self._invert is True:
                    if variable[0] != '-':
                        self.logger.error("The first argument is a deselect {}. And the current argument {} is a select".format(self._required_variables[0], variable))
                        raise ValueError("It appears that there are some select and deselect variables by the user. Note you can either select or delect variables at a time not both")

                    if variable[1:] not in self._variables_of_interest:
                        self.logger.error("The variable {} is not in {}".format(variable, self._variables_of_interest))
                        raise ValueError("The variable selected by the User is not a confounder, Instrument Variable or a Effect Modifier")
                else:
                    if variable[0] == '-':
                        self.logger.error("The first argument is a select {}. And the current argument {} is a deselect".format(self._required_variables[0], variable))
                        raise ValueError("It appears that there are some select and deselect variables by the user. Note you can either select or delect variables at a time not both") 

                    if variable not in self._variables_of_interest:
                        self.logger.error("The variable {} is not in {}".format(variable, self._variables_of_interest))
                        raise ValueError("The variable selected by the User is not a confounder, Instrument Variable or a Effect Modifier")    
        
        elif self._required_variables is None:
            self.logger.info("No required variable. Resorting to Default Behavior: Run bootstrapping without any change to the original data.")

        else:
            self.logger.warning("Incorrect type: {}. Expected an int or list".format( type(self._required_variables) ) )
            self._required_variables = None

        
        self.choose_desired_variables()

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
                        new_data[variable] += np.random.randn(self._sample_size) * self._noise * scaling_factor
                    
                    elif 'bool' in new_data[variable].dtype.name:
                        probs = np.random.uniform(0, 1, self._sample_size )
                        new_data[variable] = np.where(probs < self._probability_of_change, 
                                                        np.logical_not(new_data[variable]), 
                                                        new_data[variable]) 
                    
                    elif 'category' in new_data[variable].dtype.name:
                        self.categories = new_data[variable].unique()
                        # Find the set difference for each row
                        changed_data = new_data[variable].apply(self.set_diff)
                        # Choose one out of the remaining
                        changed_data = changed_data.apply(self.choose_random)
                        new_data[variable] = np.where(probs < self._probability_of_change, changed_data)
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
            random.shuffle(self._variables_of_interest)
            self._chosen_variables = self._variables_of_interest[:self._required_variables]
        elif type(self._required_variables) is list:
            if self._invert is False:
                self._chosen_variables = self._required_variables
            else:
                self._chosen_variables = list( set(self._variables_of_interest) - set(self._required_variables) )

        if self._chosen_variables is None:
            self.logger.info("INFO: There are no chosen variables")
        else:    
            self.logger.info("INFO: The chosen variables are: " +
                            ",".join(self._chosen_variables))

    def set_diff(self, row):
        self.categories = set(self.categories)
        return list( self.categories - set([row]) )

    def choose_random(self, row):
        return random.choice(row)