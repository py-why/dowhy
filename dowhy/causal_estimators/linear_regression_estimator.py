import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
import itertools

from dowhy.causal_estimators.regression_estimator import RegressionEstimator

class LinearRegressionEstimator(RegressionEstimator):
    """Compute effect of treatment using linear regression.

    Fits a regression model for estimating the outcome using treatment(s) and confounders. For a univariate treatment, the treatment effect is equivalent to the coefficient of the treatment variable.

    Simple method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("INFO: Using Linear Regression Estimator")
        self._linear_model = self.model

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        return expr

    def _build_model(self):
        features = self._build_features()
        model = sm.OLS(self._outcome, features).fit()
        return (features, model)

    def _estimate_confidence_intervals(self, confidence_level,
            method=None):
        conf_ints = self.model.conf_int(alpha=1-confidence_level)
        return conf_ints.to_numpy()[1:(len(self._treatment_name)+1),:]

    def _estimate_std_error(self, method=None):
        std_error = self.model.bse[1:(len(self._treatment_name)+1)]
        return std_error.to_numpy()

    def _test_significance(self, estimate_value, method=None):
        pvalue = self.model.pvalues[1:(len(self._treatment_name)+1)]
        return {'p_value': pvalue.to_numpy()} 
