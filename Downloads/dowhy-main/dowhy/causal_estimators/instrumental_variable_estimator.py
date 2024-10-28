from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sympy as sp
import sympy.stats as spstats
from statsmodels.sandbox.regression.gmm import IV2SLS

from dowhy.causal_estimator import CausalEstimate, CausalEstimator, RealizedEstimand
from dowhy.causal_identifier import IdentifiedEstimand
from dowhy.utils.api import parse_state


class InstrumentalVariableEstimator(CausalEstimator):
    """Compute effect of treatment using the instrumental variables method.

    This is also a superclass that can be inherited by other specific methods.

    Supports additional parameters as listed below.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        iv_instrument_name: Optional[Union[List, Dict, str]] = None,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        **kwargs,
    ):
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param iv_instrument_name: Name of the specific instrumental variable
            to be used. Needs to be one of the IVs identified in the
            identification step. Default is to use all the IV variables
            from the identification step.
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
            iv_instrument_name=iv_instrument_name,
            **kwargs,
        )
        self.iv_instrument_name = iv_instrument_name
        self.logger.info("INFO: Using Instrumental Variable Estimator")

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

        self.estimating_instrument_names = self._target_estimand.instrumental_variables
        if self.iv_instrument_name is not None:
            self.estimating_instrument_names = parse_state(self.iv_instrument_name)
        self.logger.debug("Instrumental Variables used:" + ",".join(self.estimating_instrument_names))
        if not self.estimating_instrument_names:
            raise ValueError("No valid instruments found. IV Method not applicable")
        if len(self.estimating_instrument_names) < len(self._target_estimand.treatment_variable):
            # TODO move this to the identification step
            raise ValueError(
                "Number of instruments fewer than number of treatments. 2SLS requires at least as many instruments as treatments."
            )
        self._estimating_instruments = data[self.estimating_instrument_names]

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        **_,
    ):
        """
        data: dataframe containing the data on which treatment effect is to be estimated.
        treatment_value: value of the treatment variable for which the effect is to be estimated.
        control_value: value of the treatment variable that denotes its absence (usually 0)
        target_units: The units for which the treatment effect should be estimated.
                     It can be a DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
                     It can also be a lambda function that can be used as an index for the data (pandas DataFrame) to select the required rows.
        """
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        if len(self.estimating_instrument_names) == 1 and len(self._target_estimand.treatment_variable) == 1:
            instrument = self._estimating_instruments.iloc[:, 0]
            self.logger.debug("Instrument Variable values: {0}".format(instrument))
            num_unique_values = len(np.unique(instrument))
            instrument_is_binary = num_unique_values <= 2
            if instrument_is_binary:
                # Obtain estimate by Wald Estimator
                y1_z = np.mean(data[self._target_estimand.outcome_variable[0]][instrument == 1])
                y0_z = np.mean(data[self._target_estimand.outcome_variable[0]][instrument == 0])
                x1_z = np.mean(data[self._target_estimand.treatment_variable[0]][instrument == 1])
                x0_z = np.mean(data[self._target_estimand.treatment_variable[0]][instrument == 0])
                num = y1_z - y0_z
                deno = x1_z - x0_z
                iv_est = num / deno
            else:
                # Obtain estimate by 2SLS estimator: Cov(y,z) / Cov(x,z)
                num_yz = np.cov(data[self._target_estimand.outcome_variable[0]], instrument)[0, 1]
                deno_xz = np.cov(data[self._target_estimand.treatment_variable[0]], instrument)[0, 1]
                iv_est = num_yz / deno_xz
        else:
            # More than 1 instrument. Use 2sls.
            est_treatment = data[self._target_estimand.treatment_variable].astype(np.float32)
            est_outcome = data[self._target_estimand.outcome_variable[0]].astype(np.float32)
            ivmodel = IV2SLS(est_outcome, est_treatment, self._estimating_instruments)
            reg_results = ivmodel.fit()
            self.logger.debug(reg_results.summary())
            iv_est = sum(
                reg_results.params
            )  # the effect is the same for any treatment value (assume treatment goes from 0 to 1)
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=iv_est,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        sym_outcome = spstats.Normal(",".join(estimand.outcome_variable), 0, 1)
        sym_treatment = spstats.Normal(",".join(estimand.treatment_variable), 0, 1)
        sym_instrument = sp.Symbol(",".join(self.estimating_instrument_names))
        sym_outcome_derivative = sp.Derivative(sym_outcome, sym_instrument)
        sym_treatment_derivative = sp.Derivative(sym_treatment, sym_instrument)
        sym_effect = spstats.Expectation(sym_outcome_derivative) / sp.stats.Expectation(sym_treatment_derivative)
        estimator_assumptions = {
            "treatment_effect_homogeneity": (
                "Each unit's treatment {0} is ".format(self._target_estimand.treatment_variable)
                + "affected in the same way by common causes of "
                "{0} and {1}".format(self._target_estimand.treatment_variable, self._target_estimand.outcome_variable)
            ),
            "outcome_effect_homogeneity": (
                "Each unit's outcome {0} is ".format(self._target_estimand.outcome_variable)
                + "affected in the same way by common causes of "
                "{0} and {1}".format(self._target_estimand.treatment_variable, self._target_estimand.outcome_variable)
            ),
        }
        sym_assumptions = {**estimand.estimands["iv"]["assumptions"], **estimator_assumptions}

        symbolic_estimand = RealizedEstimand(estimand, estimator_name="Wald Estimator")
        symbolic_estimand.update_assumptions(sym_assumptions)
        symbolic_estimand.update_estimand_expression(sym_effect)
        return symbolic_estimand
