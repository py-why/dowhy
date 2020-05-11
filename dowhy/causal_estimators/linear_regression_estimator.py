import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
import itertools

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator

class LinearRegressionEstimator(CausalEstimator):
    """Compute effect of treatment using linear regression.

    Fits a regression model for estimating the outcome using treatment(s) and confounders. For a univariate treatment, the treatment effect is equivalent to the coefficient of the treatment variable.

    Simple method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))
        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        if len(self._observed_common_causes_names)>0:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = None
        self.logger.info("INFO: Using Linear Regression Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        self._linear_model = None

    def _estimate_effect(self, data_df=None, need_conditional_estimates=None):
        # TODO make treatment_value and control value also as local parameters
        if data_df is None:
            data_df = self._data
        if need_conditional_estimates is None:
            need_conditional_estimates = bool(self._effect_modifier_names)
        # Checking if the model is already trained
        if not self._linear_model:
            # The model is always built on the entire data
            features, self._linear_model = self._build_linear_model()
            coefficients = self._linear_model.params[1:] # first coefficient is the intercept
            self.logger.debug("Coefficients of the fitted linear model: " +
                          ",".join(map(str, coefficients)))
            print(self._linear_model.summary())
        # All treatments are set to the same constant value
        effect_estimate = self._do(self._treatment_value, data_df) - self._do(self._control_value, data_df)
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                    self._estimate_effect_fn,
                    effect_modifier_names=self._effect_modifier_names)
        intercept_parameter = self._linear_model.params[0]
        estimate = CausalEstimate(estimate=effect_estimate,
                              conditional_estimates=conditional_effect_estimates,
                              target_estimand=self._target_estimand,
                              realized_estimand_expr=self.symbolic_estimator,
                              intercept=intercept_parameter)
        return estimate

    def _estimate_effect_fn(self, data_df):
        est = self._estimate_effect(data_df, need_conditional_estimates=False)
        return est.value

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        return expr 

    def _build_features(self, treatment_values=None, data_df=None):
        # Using all data by default
        if data_df is None:
            data_df = self._data
            treatment_vals = self._treatment
            observed_common_causes_vals = self._observed_common_causes
            effect_modifiers_vals = self._effect_modifiers
        else:
            treatment_vals = data_df[self._treatment_name]
            if len(self._observed_common_causes_names)>0:
                observed_common_causes_vals = data_df[self._observed_common_causes_names]
                observed_common_causes_vals = pd.get_dummies(observed_common_causes_vals, drop_first=True)
            if self._effect_modifier_names:
                effect_modifiers_vals =  data_df[self._effect_modifier_names]
                effect_modifiers_vals = pd.get_dummies(effect_modifiers_vals, drop_first=True)
        # Fixing treatment value to the specified value, if provided
        if treatment_values is not None:
            treatment_vals = treatment_values
        if type(treatment_vals) is not np.ndarray:
            treatment_vals = treatment_vals.to_numpy()
        # treatment_vals and data_df should have same number of rows
        if treatment_vals.shape[0] != data_df.shape[0]:
            raise ValueError("Provided treatment values and dataframe should have the same length.")
        # Bulding the feature matrix
        n_samples = treatment_vals.shape[0]
        treatment_2d = treatment_vals.reshape((n_samples,len(self._treatment_name)))
        if len(self._observed_common_causes_names)>0:
            features = np.concatenate((treatment_2d, observed_common_causes_vals),
                                  axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:,i]
                new_features = curr_treatment[:, np.newaxis] * effect_modifiers_vals.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(float, copy=False) # converting to float in case of binary treatment and no other variables
        features = sm.add_constant(features, has_constant='add') # to add an intercept term
        return features

    def _build_linear_model(self):
        features = self._build_features()
        model = sm.OLS(self._outcome, features).fit()
        return (features, model)

    def _do(self, treatment_val, data_df=None):
        if data_df is None:
            data_df = self._data
        if not self._linear_model:
            # The model is always built on the entire data
            _, self._linear_model = self._build_linear_model()
        # Replacing treatment values by given x
        interventional_treatment_2d = np.full((data_df.shape[0], len(self._treatment_name)), treatment_val)
        new_features = self._build_features(treatment_values=interventional_treatment_2d,
                data_df=data_df)
        interventional_outcomes = self._linear_model.predict(new_features)
        return interventional_outcomes.mean()

    def _estimate_confidence_intervals(self, confidence_level,
            method=None):
        conf_ints = self._linear_model.conf_int(alpha=1-confidence_level)
        return conf_ints.to_numpy()[1:(len(self._treatment_name)+1),:]

    def _estimate_std_error(self, method=None):
        std_error = self._linear_model.bse[1:(len(self._treatment_name)+1)]
        return std_error.to_numpy()

    def _test_significance(self, estimate_value, method=None):
        pvalue = self._linear_model.pvalues[1:(len(self._treatment_name)+1)]
        return {'p_value': pvalue.to_numpy()} 
