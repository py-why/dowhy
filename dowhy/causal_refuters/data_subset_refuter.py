from dowhy.causal_refuter import CausalRefuter, CausalRefutation
import numpy as np
import logging

class DataSubsetRefuter(CausalRefuter):
    """Refute an estimate by rerunning it on a random subset of the original data.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - 'subset_fraction': float, 0.8 by default
    Fraction of the data to be used for re-estimation.
    - 'num_of_simulations': int, 200 by default
    The number of simulations to be run
    - random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. If we with to repeat the
    same behavior we push the same seed in the psuedo-random generator
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subset_fraction = kwargs.pop("subset_fraction", 0.8)
        self._num_of_simulations = kwargs.pop("num_of_simulations", 200)
        self._random_state = kwargs.pop("random_state",None)

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def refute_estimate(self):

        sample_estimates = np.zeros(self._num_of_simulations)
        self.logger.info("Subset Fraction:{}\nNumber of Samples:{}"
                         .format(self._subset_fraction
                         ,self._num_of_simulations)
                        )

        for index in range(self._num_of_simulations):
            if self._random_state is None:
                new_data = self._data.sample(frac=self._subset_fraction)
            else:
                new_data = self._data.sample(frac=self._subset_fraction,
                                            random_state=self._random_state)
                                            
            new_estimator = self.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            sample_estimates[index] = new_effect.value

        refute = CausalRefutation(
            self._estimate.value,
            np.mean(sample_estimates),
            refutation_type="Refute: Use a subset of data"
        )
        return refute
