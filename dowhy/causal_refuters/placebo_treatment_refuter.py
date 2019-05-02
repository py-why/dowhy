import copy

import numpy as np

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class PlaceboTreatmentRefuter(CausalRefuter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._placebo_type = kwargs["placebo_type"]

    def refute_estimate(self):
        num_rows = self._data.shape[0]
        if self._placebo_type == "permute":
            new_treatment = self._data[self._treatment_name].sample(frac=1).values
        else:
            new_treatment = np.random.randn(num_rows)
        new_data = self._data.assign(placebo=new_treatment)

        self.logger.debug(new_data[0:10])
        estimator_class = self._estimate.params['estimator_class']
        identified_estimand = copy.deepcopy(self._target_estimand)
        identified_estimand.treatment_variable = ["placebo"]

        new_estimator = estimator_class(
            new_data,
            identified_estimand,
            identified_estimand.treatment_variable, self._outcome_name,
            test_significance=None
        )
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Use a Placebo Treatment")
        return refute
