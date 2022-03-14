import statsmodels.api as sm
import itertools

from dowhy.causal_estimators.regression_estimator import RegressionEstimator


class GeneralizedLinearModelEstimator(RegressionEstimator):
    """Compute effect of treatment using a generalized linear model such as logistic regression.

    Implementation uses statsmodels.api.GLM.
    Needs an additional parameter, "glm_family" to be specified in method_params. The value of this parameter can be any valid statsmodels.api families object. For example, to use logistic regression, specify "glm_family" as statsmodels.api.families.Binomial().

    """

    def __init__(self, *args, glm_family=None, predict_score=True, **kwargs):
        """For a list of args and kwargs, see documentation for
        :class:`~dowhy.causal_estimator.CausalEstimator`.

        :param glm_family: statsmodels family for the generalized linear model.
            For example, use statsmodels.api.families.Binomial() for logistic
            regression or statsmodels.api.families.Poisson() for count data.
        :param predict_score: For models that have a binary output, whether
            to output the model's score or the binary output based on the score.

        """
        # Required to ensure that self.method_params contains all the
        # parameters needed to create an object of this class
        args_dict = {k: v for k, v in locals().items()
                     if k not in type(self)._STD_INIT_ARGS}
        args_dict.update(kwargs)
        super().__init__(*args, **args_dict)
        self.logger.info("INFO: Using Generalized Linear Model Estimator")
        if glm_family is not None:
            self.family = glm_family
        else:
            raise ValueError("Need to specify the family for the generalized linear model. Provide a 'glm_family' parameter in method_params, such as statsmodels.api.families.Binomial() for logistic regression.")
        self.predict_score = predict_score
        # Checking if Y is binary
        outcome_values = self._data[self._outcome_name].astype(int).unique()
        self.outcome_is_binary = all([v in [0,1] for v in outcome_values])

    def _build_model(self):
        features = self._build_features()
        model = sm.GLM(self._outcome, features, family=self.family).fit()
        return (features, model)

    def predict_fn(self, model, features):
        if self.outcome_is_binary:
            if self.predict_score:
                return model.predict(features)
            else:
                return (model.predict(features) > 0.5).astype(int)
        else:
            return model.predict(features)

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~" + "Sigmoid("
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        expr += ")"
        return expr
