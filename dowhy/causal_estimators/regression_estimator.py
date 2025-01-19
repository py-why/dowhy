from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from dowhy.causal_estimator import CausalEstimate, CausalEstimator, IdentifiedEstimand


class RegressionEstimator(CausalEstimator):
    """Compute effect of treatment using some regression function.

    Fits a regression model for estimating the outcome using treatment(s) and
    confounders.

    Base class for all regression models, inherited by
    LinearRegressionEstimator and GeneralizedLinearModelEstimator.

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

        self.model = None

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

        self.logger.debug("Adjustment set variables used:" + ",".join(self._target_estimand.get_adjustment_set()))
        self._observed_common_causes_names = self._target_estimand.get_adjustment_set()
        if len(self._observed_common_causes_names) > 0:
            self._observed_common_causes = data[self._observed_common_causes_names]
            self._observed_common_causes = self._encode(self._observed_common_causes, "observed_common_causes")
        else:
            self._observed_common_causes = None

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        # The model is always built on the entire data
        _, self.model = self._build_model(data)
        coefficients = self.model.params[1:]  # first coefficient is the intercept
        self.logger.debug("Coefficients of the fitted model: " + ",".join(map(str, coefficients)))
        self.logger.debug(self.model.summary())

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        need_conditional_estimates=None,
        **_,
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        if need_conditional_estimates is None:
            need_conditional_estimates = self.need_conditional_estimates
        # TODO make treatment_value and control value also as local parameters
        # All treatments are set to the same constant value
        effect_estimate = self._do(treatment_value, data) - self._do(control_value, data)
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                data, self._estimate_effect_fn, effect_modifier_names=self._effect_modifier_names
            )
        intercept_parameter = self.model.params[0]
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=effect_estimate,
            control_value=control_value,
            treatment_value=treatment_value,
            conditional_estimates=conditional_effect_estimates,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            intercept=intercept_parameter,
        )

        estimate.add_estimator(self)
        return estimate

    def _estimate_effect_fn(self, data_df):
        est = self.estimate_effect(data=data_df, need_conditional_estimates=False)
        return est.value

    def _build_features(self, data_df: pd.DataFrame, treatment_values=None):
        treatment_vals = self._encode(data_df[self._target_estimand.treatment_variable], "treatment")

        if len(self._observed_common_causes_names) > 0:
            observed_common_causes_vals = data_df[self._observed_common_causes_names]
            observed_common_causes_vals = self._encode(observed_common_causes_vals, "observed_common_causes")

        if self._effect_modifier_names:
            effect_modifiers_vals = data_df[self._effect_modifier_names]
            effect_modifiers_vals = self._encode(effect_modifiers_vals, "effect_modifiers")

        # Fixing treatment value to the specified value, if provided
        if treatment_values is not None:
            treatment_vals = treatment_values
        if type(treatment_vals) is not np.ndarray:
            treatment_vals = treatment_vals.to_numpy()
        # treatment_vals and data_df should have same number of rows
        if treatment_vals.shape[0] != data_df.shape[0]:
            raise ValueError("Provided treatment values and dataframe should have the same length.")

        # Bulding the feature matrix
        n_treatment_cols = 1 if len(treatment_vals.shape) == 1 else treatment_vals.shape[1]
        n_samples = treatment_vals.shape[0]
        treatment_2d = treatment_vals.reshape((n_samples, n_treatment_cols))
        if len(self._observed_common_causes_names) > 0:
            features = np.concatenate((treatment_2d, observed_common_causes_vals), axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:, i]
                new_features = curr_treatment[:, np.newaxis] * effect_modifiers_vals.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(
            float, copy=False
        )  # converting to float in case of binary treatment and no other variables
        features = sm.add_constant(features, has_constant="add")  # to add an intercept term
        return features

    def interventional_outcomes(self, data_df: pd.DataFrame, treatment_val):
        """
        Applies an intervention treatment_val to all rows in data_df, then uses self.model
        to predict outcomes. If data_df is None, will use self._data instead.
        If no model exists, one will be created. The outcomes of all samples are returned,
        allowing analysis of individual predictions in counterfactual treatment scenarios.
        :param data_df: data frame containing the data
        :param treatment_val: value for the treatment variable
        :returns: A list of outcome predictions.
        """

        if data_df is None:
            data_df = self._data.copy()
        else:
            data_df = data_df.copy()  # don't modify arg

        # Replace treatment values with value supplied; note: Don't change column datatype!
        original_type = data_df[self._target_estimand.treatment_variable].dtypes
        data_df[self._target_estimand.treatment_variable] = treatment_val
        data_df[self._target_estimand.treatment_variable] = data_df[self._target_estimand.treatment_variable].astype(
            original_type, copy=False
        )

        return self.predict(data_df)

    def predict(self, data_df):
        if not self.model:
            # The model is always built on the entire data
            _, self.model = self._build_model()

        new_features = self._build_features(data_df=data_df)
        interventional_outcomes = self.predict_fn(data_df, self.model, new_features)
        return interventional_outcomes

    def _do(
        self,
        treatment_val,
        data_df: pd.DataFrame,
    ):
        interventional_outcomes = self.interventional_outcomes(data_df, treatment_val)
        return interventional_outcomes.mean()
