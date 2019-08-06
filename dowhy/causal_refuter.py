import logging
import numpy as np


class CausalRefuter:

    def __init__(self, data, identified_estimand, estimate, **kwargs):
        self._data = data
        self._target_estimand = identified_estimand
        self._estimate = estimate
        self._treatment_name = self._target_estimand.treatment_variable
        self._outcome_name = self._target_estimand.outcome_variable
        self._random_seed = None
        if "random_seed" in kwargs:
            self._random_seed = kwargs['random_seed']
            np.random.seed(self._random_seed)
        self.logger = logging.getLogger(__name__)


class CausalRefutation:

    def __init__(self, estimated_effect, new_effect, refutation_type):
        self.estimated_effect = estimated_effect,
        self.new_effect = new_effect,
        self.refutation_type = refutation_type

    def __str__(self):
        return "{0}\nEstimated effect:{1}\nNew effect:{2}\n".format(
            self.refutation_type, self.estimated_effect, self.new_effect
        )
