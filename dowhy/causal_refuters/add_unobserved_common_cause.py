import copy

import numpy as np
import pandas as pd

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class AddUnobservedCommonCause(CausalRefuter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refute_estimate(self):
        num_rows = self._data.shape[0]
        w_random=np.random.randn(num_rows)
        kappa = 500
        new_data = copy.deepcopy(self._data)
        new_data[self._treatment_name] = new_data[self._treatment_name].update(new_data[self._treatment_name])#pd.Series(kappa * w_random))
        new_data[self._outcome_name] = new_data[self._outcome_name].update(new_data[self._outcome_name] - pd.Series(kappa * w_random))

        estimator_class = self._estimate.params['estimator_class']
        # identified_estimand = IdentifiedEstimand(
        #        treatment_variable = self._treatment_name,
        #        outcome_variable = self._outcome_name,
        #        backdoor_variables = new_backdoor_variables)#self._target_estimand.backdoor_variables)#new_backdoor_variables)
        new_estimator = estimator_class(
            new_data,
            self._target_estimand,
            self._treatment_name, self._outcome_name,
            test_significance=None
        )
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Add an Unobserved Common Cause")
        return refute
