import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator


class RegressionDiscontinuityEstimator(CausalEstimator):
    """Compute effect of treatment using the regression discontinuity method.

    Estimates effect by transforming the problem to an instrumental variables
    problem.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """

    def __init__(self, *args, rd_variable_name=None,
            rd_threshold_value=None, rd_bandwidth=None, **kwargs):
        """
        :param rd_variable_name: Name of the variable on which the
            discontinuity occurs. This is the instrument.
        :param rd_threshold_value: Threshold at which the discontinuity occurs.
        :param rd_bandwidth: Distance from the threshold within which
            confounders can be considered the same between treatment and
            control. Considered band is (threshold +- bandwidth)

        """
        # Required to ensure that self.method_params contains all the information
        # to create an object of this class
        args_dict = {k: v for k, v in locals().items()
                     if k not in type(self)._STD_INIT_ARGS}
        args_dict.update(kwargs)
        super().__init__(*args, **args_dict)
        self.logger.info("Using Regression Discontinuity Estimator")
        self.rd_variable_name = rd_variable_name
        self.rd_threshold_value = rd_threshold_value
        self.rd_bandwidth = rd_bandwidth
        self.rd_variable = self._data[self.rd_variable_name]

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        upper_limit = self.rd_threshold_value + self.rd_bandwidth
        lower_limit = self.rd_threshold_value - self.rd_bandwidth
        rows_filter = np.s_[(self.rd_variable >= lower_limit) & (self.rd_variable <= upper_limit)]
        local_rd_variable = self.rd_variable[rows_filter]
        local_treatment_variable = self._treatment[self._treatment_name[0]][rows_filter] # indexing by treatment name again since this method assumes a single-dimensional treatment
        local_outcome_variable = self._outcome[rows_filter]
        local_df = pd.DataFrame(data={
            'local_rd_variable': local_rd_variable,
            'local_treatment': local_treatment_variable,
            'local_outcome': local_outcome_variable
        })
        print(local_df)
        iv_estimator = InstrumentalVariableEstimator(
            local_df,
            self._target_estimand,
            ['local_treatment'],
            ['local_outcome'],
            test_significance=self._significance_test,
            iv_instrument_name='local_rd_variable'
        )
        est = iv_estimator.estimate_effect()
        return est

    def construct_symbolic_estimator(self, estimand):
        return ""
