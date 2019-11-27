from dowhy.causal_refuter import CausalRefuter, CausalRefutation


class DataSubsetRefuter(CausalRefuter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subset_fraction = kwargs["subset_fraction"]

    def refute_estimate(self):
        new_data = self._data.sample(frac=self._subset_fraction)

        new_estimator = self.get_estimator_object(new_data, self._target_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()

        refute = CausalRefutation(
            self._estimate.value,
            new_effect.value,
            refutation_type="Refute: Use a subset of data"
        )
        return refute
