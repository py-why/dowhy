import numpy as np
import pandas as pd

import SparseSC

from dowhy.causal_estimator import CausalEstimate, CausalEstimator

class SyntheticControlsEstimator(CausalEstimator):
    """Compute effect of treatment using Synthetic Controls method"""

    def __init__(self, *args, **kwargs):
        """For a list of standard args and kwargs, see documentation for
        :class:`~dowhy.causal_estimator.CausalEstimator`.

        """

        super().__init__(*args, **kwargs)


    def _estimate_effect(self, data_df=None):



    def _estimate_effect_fn(self, data_df):


    def _do(self, treatment_val, data_df=None):