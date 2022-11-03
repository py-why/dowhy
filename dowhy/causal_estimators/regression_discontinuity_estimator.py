from typing import Any, List, Optional

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

    def __init__(
        self,
        identified_estimand,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        num_null_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction=CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level=CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates="auto",
        num_quantiles_to_discretize_cont_cols=CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        rd_variable_name=None,
        rd_threshold_value=None,
        rd_bandwidth=None,
        **kwargs,
    ):
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
        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            rd_variable_name=rd_variable_name,
            rd_threshold_value=rd_threshold_value,
            rd_bandwidth=rd_bandwidth,
            **kwargs,
        )
        self.logger.info("Using Regression Discontinuity Estimator")
        self.rd_variable_name = rd_variable_name
        self.rd_threshold_value = rd_threshold_value
        self.rd_bandwidth = rd_bandwidth

    def fit(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        self.set_data(data, treatment_name, outcome_name)
        self.set_effect_modifiers(effect_modifier_names)

        self.rd_variable = self._data[self.rd_variable_name]

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def estimate_effect(self, treatment_value: Any = 1, control_value: Any = 0, target_units=None, **_):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        upper_limit = self.rd_threshold_value + self.rd_bandwidth
        lower_limit = self.rd_threshold_value - self.rd_bandwidth
        rows_filter = np.s_[(self.rd_variable >= lower_limit) & (self.rd_variable <= upper_limit)]
        local_rd_variable = self.rd_variable[rows_filter]
        local_treatment_variable = self._treatment[self._treatment_name[0]][
            rows_filter
        ]  # indexing by treatment name again since this method assumes a single-dimensional treatment
        local_outcome_variable = self._outcome[rows_filter]
        local_df = pd.DataFrame(
            data={
                "local_rd_variable": local_rd_variable,
                "local_treatment": local_treatment_variable,
                "local_outcome": local_outcome_variable,
            }
        )
        self.logger.debug(local_df)
        iv_estimator = InstrumentalVariableEstimator(
            self._target_estimand,
            test_significance=self._significance_test,
        )

        iv_estimator.fit(
            local_df,
            ["local_treatment"],
            ["local_outcome"],
            iv_instrument_name="local_rd_variable",
        )

        est = iv_estimator.estimate_effect()

        est.add_estimator(self)
        return est

    def construct_symbolic_estimator(self, estimand):
        return ""
