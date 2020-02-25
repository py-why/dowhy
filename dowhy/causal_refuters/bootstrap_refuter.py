from dowhy.causal_refuter import CausalRefuter, CausalRefutation
from sklearn.utils import resample

class BootstrapRefuter(CausalRefuter):
    """
    Refute an estimate by running it on a random sample of the original data.
    It supports additional parameters that can be specified in the regute_estimate() method.
    - 'number_of_sumples': int, None by default
    The number of bootstrap samples to be constructed
    - 'random_state': int, RandomState, None by default
    The seed value to be added if we wish to repeat the same random behavior. For this purpose, 
    we repeat the same seed in the psuedo-random generator.
    """

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self._number_of_samples = options.pop("number_of_samples", 200)
        self._random_state = options.pop("random_state",None)
        
    def refute_estimate(self, *args, **kwargs):
        if self._random_state is None:
            new_data = resample(self._data, 
                                n_samples=self._number_of_samples )
        else:
            new_data = resample(self._data,
                                n_samples=self._number_of_samples,
                                random_state=self._random_state)
            
            sample_estimates = np.zeros(self._number_of_samples) 
            for index in range( len(new_data) ):
                new_estimator = self.get_estimator_object(new_data, self._target_estimand, self._estimate)
                new_effect = new_estimator.estimate_effect()
                sample_estimates[i] = new_effect.value

            refute = CausalRefutation(
                self._estimate.value,
                np.mean(sample_estimates),
                refutation_type="Refute: Bootstrap Sample Dataset"
            )

            return refute

