import logging
import numpy as np


class CausalRefuter:
    
    """Base class for different refutation methods. 

    Subclasses implement specific refutations methods. 

    """

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

    @staticmethod
    def get_estimator_object(new_data, identified_estimand, estimate):
        estimator_class = estimate.params['estimator_class']
        new_estimator = estimator_class(
                new_data,
                identified_estimand,
                identified_estimand.treatment_variable, identified_estimand.outcome_variable, #names of treatment and outcome
                test_significance=None,
                evaluate_effect_strength=False,
                confidence_intervals = estimate.params["confidence_intervals"],
                target_units = estimate.params["target_units"],
                effect_modifiers = estimate.params["effect_modifiers"],
                params = estimate.params["method_params"]
                )
        return new_estimator

    def refute_estimate(self):
        raise NotImplementedError


class CausalRefutation:
    """Class for storing the result of a refutation method.

    """

    def __init__(self, estimated_effect, new_effect, refutation_type):
        self.estimated_effect = estimated_effect,
        self.new_effect = new_effect,
        self.refutation_type = refutation_type

    def __str__(self):
        return "{0}\nEstimated effect:{1}\nNew effect:{2}\n".format(
            self.refutation_type, self.estimated_effect, self.new_effect
        )
