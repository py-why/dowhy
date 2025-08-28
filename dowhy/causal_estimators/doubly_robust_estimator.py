from typing import Any, List, Optional, Type, Union

import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class DoublyRobustEstimator(CausalEstimator):
    """Doubly Robust Estimator for Causal Effect Estimation.

    Supports any RegressionEstimator for the regression stage, and accepts
    a propensity score model and column for the propensity score stage.

    References
    ----------
    [1] Michele Jonsson Funk, Daniel Westreich, Chris Wiesen, Til Stürmer,
        M. Alan Brookhart, Marie Davidian, Doubly Robust Estimation of Causal
        Effects, American Journal of Epidemiology, Volume 173, Issue 7,
        1 April 2011, Pages 761-767, https://doi.org/10.1093/aje/kwq439
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
        regression_estimator: Union[RegressionEstimator, Type[RegressionEstimator]] = LinearRegressionEstimator,
        propensity_score_model: Optional[Any] = None,
        propensity_score_column: str = "propensity_score",
        min_ps_score: float = 0.01,
        max_ps_score: float = 0.99,
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
            estimation of conditional treatment effect over it
        :param regression_estimator: RegressionEstimator used for the regression
            stage of the doubly robust formula. Can be any class that extends the
            RegressionEstimator class. Default='LinearRegressionEstimator'
        :param propensity_score_model: Model used to compute propensity score.
            Can be any classification model that supports fit() and
            predict_proba() methods. If None, LogisticRegression is used
        :param propensity_score_column: Column name that stores the
            propensity score. Default='propensity_score'
        :param min_ps_score: Lower bound used to clip the propensity score.
            Default=0.01
        :param max_ps_score: Upper bound used to clip the propensity score.
            Default=0.99
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
            **kwargs,
        )
        # Initialize the subcomponents
        self.regression_model = (
            regression_estimator
            if isinstance(regression_estimator, RegressionEstimator)
            else regression_estimator(
                identified_estimand=identified_estimand,
                **kwargs,
            )
        )
        self.propensity_score_model = PropensityScoreEstimator(
            identified_estimand=identified_estimand,
            propensity_score_model=propensity_score_model,
            propensity_score_column=propensity_score_column,
        )
        self.min_ps_score = min_ps_score
        self.max_ps_score = max_ps_score

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param effect_modifier_names: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        # Validate target estimand
        if len(self._target_estimand.treatment_variable) > 1:
            error_msg = str(self.__class__) + " cannot handle more than one treatment variable"
            raise Exception(error_msg)
        if len(self._target_estimand.outcome_variable) > 1:
            error_msg = str(self.__class__) + " cannot handle more than one outcome variable"
            raise Exception(error_msg)
        if self._target_estimand.identifier_method not in ["backdoor", "general_adjustment"]:
            error_msg = str(self.__class__) + " only supports covariate adjustment identifiers"
            raise Exception(error_msg)
        if effect_modifier_names and (len(effect_modifier_names) > 0):
            # TODO: Add support for effect modifiers in the Doubly Robust Estimator
            raise NotImplementedError("Effect Modifiers not supported yet for " + str(self.__class__))

        # Fit the models
        self._set_effect_modifiers(data, effect_modifier_names)
        self.regression_model = self.regression_model.fit(data, effect_modifier_names=effect_modifier_names)
        self.propensity_score_model = self.propensity_score_model.fit(data, effect_modifier_names=effect_modifier_names)
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        control_value: Union[float, int] = 0,
        treatment_value: Union[float, int] = 1,
        target_units: Union[str, pd.DataFrame] = "ate",
        **kwargs,
    ):
        """
        Estimate the causal effect using the Doubly Robust Formula:

        Y_{i, t}^{DR} = E[Y | X_i, T_i=t]\
            + \\frac{Y_i - E[Y | X_i, T_i=t]}{Pr[T_i=t | X_i]} \\cdot 1\\{T_i=t\\}

        Where we use our regression_model to estimate E[Y | X_i, T_i=t], and our propensity_score_model
        to estimate Pr[T_i=t | X_i].
        :param data: data frame containing the data
        :param control_value: value associated with not receiving the treatment. Default=0
        :param treatment_value: value associated with receiving the treatment. Default=1
        :param target_units: (Experimental) The units for which the treatment effect should be estimated. Eventually, this can be of three types. (1) a string for common specifications of target units (namely, "ate", "att" and "atc"), (2) a lambda function that can be used as an index for the data (pandas DataFrame), or (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data. Currently, only "ate" is supported.
        """
        if target_units != "ate":
            raise NotImplementedError("ATE is the only target unit supported for " + str(self.__class__))

        self._treatment_value = treatment_value
        self._control_value = control_value
        self._target_units = "ate"  # TODO: add support for other target units
        effect_estimate = self._do(treatment_value, treatment_value, data) - self._do(
            control_value, treatment_value, data
        )

        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=effect_estimate,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
        )
        estimate.add_estimator(self)
        return estimate

    def _do(
        self,
        treatment,
        received_treatment_value,
        data_df: pd.DataFrame,
    ):
        """
        Evaluate doubly robust model for a given treatment value.
        :param treatment: the value assigned to the treatment variable
        :param received_treatment_value: value associated with receiving the treatment
        :param data_df: data frame containing the data
        """
        # Vector representation of E[Y | X_i, T_i=t]
        regression_est_outcomes = self.regression_model.interventional_outcomes(data_df, treatment)
        # Vector representation of Y
        true_outcomes = np.array(data_df[self._target_estimand.outcome_variable[0]])
        # Vector representation of Pr[T_i=t | X_i]
        propensity_scores = np.array(
            self.propensity_score_model.predict_proba(data_df)[:, int(treatment == received_treatment_value)]
        )
        propensity_scores = np.maximum(self.min_ps_score, propensity_scores)
        propensity_scores = np.minimum(self.max_ps_score, propensity_scores)
        if propensity_scores.min() <= 0:  # Can only be reached if the caller sets min_ps_score <= 0
            raise ValueError("Propensity scores must be strictly positive for doubly robust estimation.")
        # Vector representation of 1_{T_i=t}
        treatment_indicator = np.array(data_df[self._target_estimand.treatment_variable[0]] == treatment)

        # Doubly robust formula
        outcomes = (
            regression_est_outcomes
            + (true_outcomes - regression_est_outcomes) / propensity_scores * treatment_indicator
        )
        return outcomes.mean()

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.get_adjustment_set()
        expr += "+".join(var_list)
        return expr
