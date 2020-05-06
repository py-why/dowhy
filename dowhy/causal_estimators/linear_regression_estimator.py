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

    Demo method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

    Common method but the assumptions required are too strong.

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

    def _estimate_effect(self):
        features, self._linear_model = self._build_linear_model()
        coefficients = self._linear_model.params[1:] # first coefficient is the intercept
        self.logger.debug("Coefficients of the fitted linear model: " +
                          ",".join(map(str, coefficients)))
        # All treatments are set to the same constant value
        effect_estimate = self._do(self._treatment_value) - self._do(self._control_value)
        intercept_parameter = self._linear_model.params[0]
        estimate = CausalEstimate(estimate=effect_estimate,
                              target_estimand=self._target_estimand,
                              realized_estimand_expr=self.symbolic_estimator,
                              intercept=intercept_parameter)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        return expr 

    def _build_features(self):
        n_samples = self._treatment.shape[0]
        treatment_2d = self._treatment.to_numpy().reshape((n_samples,len(self._treatment_name)))
        if len(self._observed_common_causes_names)>0:
            features = np.concatenate((treatment_2d, self._observed_common_causes),
                                  axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:,i]
                new_features = curr_treatment[:, np.newaxis] * self._effect_modifiers.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(float, copy=False) # converting to float in case of binary treatment and no other variables
        features = sm.add_constant(features) # to add an intercept term
        return features

    def _build_linear_model(self):
        features = self._build_features()
        model = sm.OLS(self._outcome, features).fit()
        return (features, model)

    def _do(self, x):
        if not self._linear_model:
            _, self._linear_model = self._build_linear_model()
        # Replacing treatment values by given x
        interventional_treatment_2d = np.full((self._treatment.shape[0], len(self._treatment_name)), x)
        features = self._build_features()
        new_features = np.concatenate((features[:,0, np.newaxis], interventional_treatment_2d, features[:,(1+len(self._treatment_name)): ]), axis=1) 
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
