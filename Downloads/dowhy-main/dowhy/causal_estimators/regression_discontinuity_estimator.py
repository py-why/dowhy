from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class RegressionDiscontinuityEstimator(CausalEstimator):
    """Compute effect of treatment using the regression discontinuity method.

    Estimates effect by transforming the problem to an instrumental variables
    problem.

    Supports additional parameters as listed below.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        rd_variable_name: Optional[str] = None,
        rd_threshold_value: Optional[float] = None,
        rd_bandwidth: Optional[float] = None,
        **kwargs,
    ):
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param test_significance: Binary flag or a string indicating whether to test significance and by which method. All estimators support test_significance="bootstrap" that estimates a p-value for the obtained estimate using the bootstrap method. Individual estimators can override this to support custom testing methods. The bootstrap method supports an optional parameter, num_null_simulations. If False, no testing is done. If True, significance of the estimate is tested using the custom method if available, otherwise by bootstrap.
        :param evaluate_effect_strength: (Experimental) whether to evaluate the strength of effect
        :param confidence_intervals: Binary flag or a string indicating whether the confidence intervals should be computed and which method should be used. All methods support estimation of confidence intervals using the bootstrap method by using the parameter confidence_intervals="bootstrap". The bootstrap method takes in two arguments (num_simulations and sample_size_fraction) that can be optionally specified in the params dictionary. Estimators may also override this to implement their own confidence interval method. If this parameter is False, no confidence intervals are computed. If True, confidence intervals are computed by the estimator's specific method if available, otherwise through bootstrap
        :param num_null_simulations: The number of simulations for testing the
            statistical significance of the estimator
        :param num_simulations: The number of simulations for finding the
            confidence interval (and/or standard error) for a estimate
        :param sample_size_fraction: The size of the sample for the bootstrap
            estimator
        :param confidence_level: The confidence level of the confidence
            interval estimate
        :param need_conditional_estimates: Boolean flag indicating whether
            conditional estimates should be computed. Defaults to True if
            there are effect modifiers in the graph
        :param num_quantiles_to_discretize_cont_cols: The number of quantiles
            into which a numeric effect modifier is split, to enable
            estimation of conditional treatment effect over it.
        :param rd_variable_name: Name of the variable on which the
            discontinuity occurs. This is the instrument.
        :param rd_threshold_value: Threshold at which the discontinuity occurs.
        :param rd_bandwidth: Distance from the threshold within which
            confounders can be considered the same between treatment and
            control. Considered band is (threshold +- bandwidth)
        :param kwargs: (optional) Additional estimator-specific parameters
        """
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
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        self.reset_encoders()  # Forget any existing encoders
        self._set_effect_modifiers(data, effect_modifier_names)

        self.rd_variable = data[self.rd_variable_name]

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        upper_limit = self.rd_threshold_value + self.rd_bandwidth
        lower_limit = self.rd_threshold_value - self.rd_bandwidth
        rows_filter = np.s_[(self.rd_variable >= lower_limit) & (self.rd_variable <= upper_limit)]
        local_rd_variable = self.rd_variable[rows_filter]
        local_treatment_variable = data[self._target_estimand.treatment_variable[0]][
            rows_filter
        ]  # indexing by treatment name again since this method assumes a single-dimensional treatment
        local_outcome_variable = data[self._target_estimand.outcome_variable[0]][rows_filter]
        self._local_df = pd.DataFrame(
            data={
                "local_rd_variable": local_rd_variable,
                self._target_estimand.treatment_variable[0]: local_treatment_variable,
                self._target_estimand.outcome_variable[0]: local_outcome_variable,
            }
        )
        self.logger.debug(self._local_df)
        self.iv_estimator = InstrumentalVariableEstimator(
            self._target_estimand,
            test_significance=self._significance_test,
            iv_instrument_name="local_rd_variable",
        )

        self.iv_estimator.fit(self._local_df)

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        **_,
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value

        est = self.iv_estimator.estimate_effect(
            self._local_df,
            treatment_value=treatment_value,
            control_value=control_value,
            target_units=target_units,
        )

        est.add_estimator(self)
        return est

    def construct_symbolic_estimator(self, estimand):
        return ""
