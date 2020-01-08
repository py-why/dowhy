import copy

import numpy as np

from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuter import CausalRefuter


class PlaceboTreatmentRefuter(CausalRefuter):
    """Refute an estimate by replacing treatment with a randomly-generated placebo variable.

    Supports additional parameters that can be specified in the refute_estimate() method.

    - '_placebo_type': Default is to generate random values for the treatment. If placebo_type is "permute", then the original treatment values are permuted by row.

    """
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

        new_estimator = self.get_estimator_object(new_data, identified_estimand, self._estimate)
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(self._estimate.value, new_effect.value,
                                  refutation_type="Refute: Use a Placebo Treatment")
        return refute
