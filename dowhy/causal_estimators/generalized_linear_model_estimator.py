import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools

from dowhy.causal_estimators.regression_estimator import RegressionEstimator

class GeneralizedLinearModelEstimator(RegressionEstimator):
    """Compute effect of treatment using a generalized linear model such as logistic regression.

    Implementation uses statsmodels.api.GLM.
    Needs an additional parameter, "glm_family" to be specified in method_params. The value of this parameter can be any valid statsmodels.api families object. For example, to use logistic regression, specify "glm_family" as statsmodels.api.families.Binomial(). 

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("INFO: Using Generalized Linear Model Estimator")
        if 'glm_family' in self.method_params:
            self.family = self.method_params['glm_family']
        else:
            raise ValueError("Need to specify the family for the generalized linear model. Provide a 'glm_family' parameter in method_params, such as statsmodels.api.families.Binomial() for logistic regression.")

    def _build_model(self):
        features = self._build_features()
        model = sm.GLM(self._outcome, features, family=self.family).fit()
        return (features, model)

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~" + "Sigmoid("
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        expr += ")"
        return expr
