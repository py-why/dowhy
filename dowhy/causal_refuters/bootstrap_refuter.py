from dowhy.causal_refuter import CausalRefuter, CausalRefutation
import numpy as np
from sklearn.utils import resample
import logging

class BootstrapRefuter(CausalRefuter):
    """
    Refute an estimate by running it on a random sample of the original data.
    It supports additional parameters that can be specified in the refute_estimate() method.
    - 'num_of_simulations': int, None by default
    The number of bootstrap simulations to be run
    - 'sample_size': int, None by default
    The size of each bootstrap sample
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. For this purpose, 
    we repeat the same seed in the psuedo-random generator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_of_simulations = kwargs.pop("num_of_simulations", 200)
        self._sample_size = kwargs.pop("sample_size",len(self._data))
        self._random_state = kwargs.pop("random_state",None)

        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def refute_estimate(self, *args, **kwargs):
        if self._sample_size > len(self._data):
                self.logger.warning("The sample size is larger than the population size")

        sample_estimates = np.zeros(self._num_of_simulations)
        self.logger.info("Sample Size:{}\nNumber of Samples:{}"
                         .format(self._sample_size
                         ,self._num_of_simulations)
                        ) 
        
        for index in range( self._num_of_simulations ):
            if self._random_state is None:
                new_data = resample(self._data, 
                                n_samples=self._sample_size )
            else:
                new_data = resample(self._data,
                                    n_samples=self._sample_size,
                                    random_state=self._random_state )

            new_estimator = self.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            sample_estimates[index] = new_effect.value

        refute = CausalRefutation(
            self._estimate.value,
            np.mean(sample_estimates),
            refutation_type="Refute: Bootstrap Sample Dataset"
        )

        return refute

