import copy

import numpy as np

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class PlaceboTreatmentRefuter(CausalRefuter):
    """Refute an estimate by replacing treatment with a randomly-generated placebo variable.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - 'placebo_type':  str, None by default
    Default is to generate random values for the treatment. If placebo_type is "permute", 
    then the original treatment values are permuted by row.
    - 'num_simulations': int, CausalRefuter.DEFAULT_NUM_SIMULATIONS by default
    The number of simulations to be run
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. If we with to repeat the
    same behavior we push the same seed in the psuedo-random generator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._placebo_type = kwargs.pop("placebo_type",None)
        if self._placebo_type is None:
            self._placebo_type = "Random Data"
        self._num_simulations = kwargs.pop("num_simulations",CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._random_state = kwargs.pop("random_state",None)

    def refute_estimate(self):

        # We need to change the identified estimand
        # This is done as a safety measure, we don't want to change the
        # original DataFrame
        identified_estimand = copy.deepcopy(self._target_estimand)
        identified_estimand.treatment_variable = ["placebo"]

        sample_estimates = np.zeros(self._num_simulations)
        self.logger.info("Refutation over {} simulated datasets of {} treatment"
                        .format(self._num_simulations
                        ,self._placebo_type)
                        )

        num_rows = self._data.shape[0]

        for index in range(self._num_simulations):

            if self._placebo_type == "permute":
                if self._random_state is None:
                    new_treatment = self._data[self._treatment_name].sample(frac=1).values
                else:
                    new_treatment = self._data[self._treatment_name].sample(frac=1, 
                                                                    random_state=self._random_state).values                    
            else:
                new_treatment = np.random.randn(num_rows)
            
            # Create a new column in the data by the name of placebo
            new_data = self._data.assign(placebo=new_treatment)

            # Sanity check the data
            self.logger.debug(new_data[0:10])

            
            new_estimator = self.get_estimator_object(new_data, identified_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            sample_estimates[index] = new_effect.value
        

        refute = CausalRefutation(self._estimate.value, 
                                  np.mean(sample_estimates),
                                  refutation_type="Refute: Use a Placebo Treatment")
                                  
        # Note: We hardcode the estimate value to ZERO as we want to check if it falls in the distribution of the refuter
        # Ideally we should expect that ZERO should fall in the distribution of the effect estimates as we have severed any causal
        # relationship between the treatment and the outcome.
        refute.add_significance_test_results(
            self.test_significance(0, sample_estimates)
        )        
        
        return refute
