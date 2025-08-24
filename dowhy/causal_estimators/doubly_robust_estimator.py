from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_identifier import IdentifiedEstimand
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator

from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator


class DoublyRobustEstimator(CausalEstimator):
    """Doubly Robust Estimator for Causal Effect Estimation."""

    # Default regression model for doubly robust estimation
    DEFAULT_REGRESSION_MODEL = LinearRegressionEstimator

    # Default propensity score model for doubly robust estimation
    DEFAULT_PROPENSITY_SCORE_MODEL = PropensityScoreEstimator

    
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
        regression_model_class: RegressionEstimator = DEFAULT_REGRESSION_MODEL,
        propensity_score_model_class: PropensityScoreEstimator = DEFAULT_PROPENSITY_SCORE_MODEL,
        regression_model_kwargs: dict = None,
        propensity_score_model_kwargs: dict = None,
        **kwargs,
    ):
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
        self.logger.info("INFO: Using Doubly Robust Estimator")
        # Initialize the models; most constructor parameters aren't necessary
        self.regression_model_class = regression_model_class
        self.regression_model_kwargs = regression_model_kwargs or {}
        self.propensity_score_model_class = propensity_score_model_class
        self.propensity_score_model_kwargs = propensity_score_model_kwargs or {}


    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """Fit the estimator with data."""
        # Check if the target estimand is valid
        if len(self._target_estimand.treatment_variable) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        if len(self._target_estimand.outcome_variable) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one outcome variable"
            raise Exception(error_msg)

        self.regression_model = self.regression_model_class(
            identified_estimand=self._target_estimand,
            **self.regression_model_kwargs,
        )
        self.propensity_score_model = self.propensity_score_model_class(
            identified_estimand=self._target_estimand,
            **self.propensity_score_model_kwargs,
        )

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
        """Estimate the causal effect using the Doubly Robust Formula."""
        self._treatment_value = treatment_value
        self._control_value = control_value
        self._target_units = target_units
        effect_estimate = self._do(treatment_value, treatment_value, data) - self._do(control_value, treatment_value, data)

        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=effect_estimate,
            control_value=control_value,
            treatment_value=treatment_value,
            conditional_estimates=None,  # TODO
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,  # TODO
        )
        estimate.add_estimator(self)
        return estimate

    
    def _do(
        self,
        treatment,
        treatment_value,
        data_df: pd.DataFrame,
    ):
        """
        Doubly Robust Formula:

        Y_{i, t}^{DR} = E[Y | X_i, T_i=t]\
            + \\frac{Y_i - E[Y | X_i, T_i=t]}{Pr[T_i=t | X_i]} \\cdot 1\\{T_i=t\\}

        Where we use our regression_model to estimate E[Y | X_i, T_i=t], and our propensity_score_model
        to estimate Pr[T_i=t | X_i].
        """

        # Vector representation of E[Y | X_i, T_i=t]
        regression_est_outcomes = self.regression_model.interventional_outcomes(data_df, treatment)
        # Vector representation of Y
        true_outcomes = np.array(data_df[self._target_estimand.outcome_variable[0]])
        # Vector representation of Pr[T_i=t | X_i]
        propensity_scores = np.array(self.propensity_score_model.predict_proba(data_df)[:, int(treatment == treatment_value)])
        # Vector representation of 1\\{T_i=t\\}
        treatment_indicator = np.array(data_df[self._target_estimand.treatment_variable[0]] == treatment)

        # Doubly robust formula
        outcomes = regression_est_outcomes + (true_outcomes - regression_est_outcomes) / propensity_scores * treatment_indicator
        return outcomes.mean()

    def construct_symbolic_estimator(self, estimand):
        return "(TODO)"
