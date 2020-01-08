import copy

import numpy as np

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class RandomCommonCause(CausalRefuter):
    """Refute an estimate by introducing a randomly generated confounder
    (that may have been unobserved).

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refute_estimate(self):
        num_rows = self._data.shape[0]
        new_data = self._data.assign(w_random=np.random.randn(num_rows))
        new_backdoor_variables = self._target_estimand.backdoor_variables + ['w_random']
        identified_estimand = copy.deepcopy(self._target_estimand)
        # Adding a new backdoor variable to the identified estimand
        identified_estimand.backdoor_variables = new_backdoor_variables

        new_estimator = self.get_estimator_object(new_data, identified_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Add a Random Common Cause")
        return refute
