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
        identified_estimand.backdoor_variables = new_backdoor_variables
        # identified_estimand = IdentifiedEstimand(
        #        treatment_variable = self._treatment_name,
        #        outcome_variable = self._outcome_name,
        #        backdoor_variables = new_backdoor_variables)#self._target_estimand.backdoor_variables)#new_backdoor_variables)
        new_estimator = estimator_class(
            new_data,
            identified_estimand,
            self._treatment_name, self._outcome_name,
            test_significance=None
        )
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Add a Random Common Cause")
        return refute
