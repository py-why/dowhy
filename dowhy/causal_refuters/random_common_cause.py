import copy

import numpy as np

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class RandomCommonCause(CausalRefuter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refute_estimate(self):
        num_rows = self._data.shape[0]
        new_data = self._data.assign(w_random=np.random.randn(num_rows))
        self.logger.debug(new_data[0:10])
        new_backdoor_variables = self._target_estimand.backdoor_variables + ['w_random']
        estimator_class = self._estimate.params['estimator_class']
        identified_estimand = copy.deepcopy(self._target_estimand)

        # Adding a new backdoor variable to the identified estimand
        identified_estimand.backdoor_variables = new_backdoor_variables
        new_estimator = estimator_class(
                new_data,
                identified_estimand,
                self._treatment_name, self._outcome_name, #names of treatment and outcome
                test_significance=None,
                evaluate_effect_strength=False,
                confidence_intervals = self._estimate.params["confidence_intervals"],
                target_units = self._estimate.params["target_units"],
                effect_modifiers = self._estimate.params["effect_modifiers"],
                params=self._estimate.params["method_params"]
                )
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Add a Random Common Cause")
        return refute
