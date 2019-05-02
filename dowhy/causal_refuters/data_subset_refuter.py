from dowhy.causal_refuter import CausalRefuter, CausalRefutation


class DataSubsetRefuter(CausalRefuter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subset_fraction = kwargs["subset_fraction"]

    def refute_estimate(self):
        new_data = self._data.sample(frac=self._subset_fraction)

        estimator_class = self._estimate.params['estimator_class']
        identified_estimand = self._target_estimand
        new_estimator = estimator_class(
            new_data,
            identified_estimand,
            self._treatment_name,
            self._outcome_name,
            test_significance=None
        )
        new_effect = new_estimator.estimate_effect()

        refute = CausalRefutation(
            self._estimate.value,
            new_effect.value,
            refutation_type="Refute: Use a subset of data"
        )
        return refute
