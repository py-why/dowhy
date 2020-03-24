import copy

import numpy as np
import logging

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter

class DummyOutcomeRefuter(CausalRefuter):
    """Refute an estimate by replacing the outcome with a randomly generated variable.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - 'dummy_outcome_type': str, None by default
    Default is to generate random values for the treatment. If palcebo_type is "permute",
    then the original treatement values are permuted by row.
    - 'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. If we want to repeat the
    same behavior we push the same seed in the psuedo-random generator
    - 'outcome_function': function pd.Dataframe -> np.ndarray, None
    A function that takes in a function that takes the input data frame as the input and outputs the outcome
    variable. This allows us to create an output varable that only depends on the confounders and does not depend 
    on the treatment variable.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dummy_outcome_type = kwargs.pop("placebo_type", None)
        if self._dummy_outcome_type is None:
            self._dummy_outcome_type = "Random Data"
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._random_state = kwargs.pop("random_state", None)
        self._outcome_function = kwargs.pop("outcome_function", None)

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        
        self.logger = logging.getLogger(__name__)

    def refute_estimate(self):

        # We need to change the identified estimand
        # We thus, make a copy. This is done as we don't want
        # to change the original DataFrame
        identified_estimand = copy.deepcopy(self._target_estimand)
        identified_estimand.outcome_variable = ["dummy_outcome"]

        sample_estimates = np.zeros(self._num_simulations)
        self.logger.info("Refutation over {} simulated datasets of {} treatment"
                        .format(self._num_simulations
                        ,self._dummy_outcome_type)
                        )
        num_rows =  self._data.shape[0]

        for index in range(self._num_simulations):

            if self._dummy_outcome_type == "permute":
                if self._random_state is None:
                    new_outcome = self._data[self._outcome_name].sample(frac=1).values
                else:
                    new_outcome = self._data[self._outcome_name].sample(frac=1,
                                                                random_state=self._random_state).values
            elif self._outcome_function is not None:
                new_outcome = self._outcome_function(self._data)
                if len(new_outcome.shape) == 2 and 
                    ( new_outcome.shape[0] ==1 or new_outcome.shape[1] ):
                    self.logger.warning("Converting the row or column vector to 1D array")
                    new_outcome = new_outcome.ravel()
                    assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
                elif len(new_oucome.shape) == 1:
                    assert len(new_outcome) == num_rows, ("The number of outputs do not match that of the number of outcomes")
                else:
                    raise Exception("Type Mismatch: The outcome is one dimensional, but the output has the shape:{}".format(new_outcome.shape))
            else:
                new_outcome = np.random.randn(num_rows)

        # Create a new column in the data by the name of dummy_outcome
        new_data = self._data.assign(dummy_outcome=new_outcome)

        # Sanity check the data
        self.logger.debug(new_data[0:10])

        new_estimator = self.get_estimator_object(new_data, identified_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()
        sample_estimates[index] = new_effect.value

        refute = CausalRefutation(self._estimate.value,
                                        np.mean(sample_estimates),
                                        refutation_type="Refute: Use a Dummy Outcome")
        
        # Note: We hardcode the estimate value to ZERO as we want to check if it falls in the distribution of the refuter
        # Ideally we should expect that ZERO should fall in the distribution of the effect estimates as we have severed any causal 
        # relationship between the treatment and the outcome.

        dummy_estimator = copy.deepcopy(self._estimate)
        dummy_estimator.value = 0

        refute.add_significance_test_results(
            self.test_significance(dummy_estimator, sample_estimates)
        )

        return refute