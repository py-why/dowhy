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
        if self._effect_modifier_names:
            # The average treatment effect is a combination of different
            # regression coefficients. Complicated to compute the confidence
            # interval analytically. For example, if y=a + b1.t + b2.tx, then
            # the average treatment effect is b1+b2.mean(x).
            # Refer Gelman, Hill. ARM Book. Chapter 9
            # http://www.stat.columbia.edu/~gelman/arm/chap9.pdf
            # TODO: Looking for contributions
            raise NotImplementedError
        else:
            conf_ints = self.model.conf_int(alpha=1-confidence_level)
            # For a linear regression model, the causal effect of a variable is equal to the coefficient corresponding to the
            # variable. Hence, the model by default outputs the confidence interval corresponding to treatment=1 and control=0.
            # So for custom treatment and control values, we must multiply the confidence interval by the difference of the two.
            return (self._treatment_value - self._control_value) * conf_ints.to_numpy()[1:(len(self._treatment_name)+1),:]

    def _estimate_std_error(self, method=None):
        if self._effect_modifier_names:
            raise NotImplementedError
        else:
            std_error = self.model.bse[1:(len(self._treatment_name)+1)]

            # For a linear regression model, the causal effect of a variable is equal to the coefficient corresponding to the
            # variable. Hence, the model by default outputs the standard error corresponding to treatment=1 and control=0.
            # So for custom treatment and control values, we must multiply the standard error by the difference of the two.
            return (self._treatment_value - self._control_value) * std_error.to_numpy()

    def _test_significance(self, estimate_value, method=None):
        pvalue = self.model.pvalues[1:(len(self._treatment_name)+1)]
        return {'p_value': pvalue.to_numpy()}
