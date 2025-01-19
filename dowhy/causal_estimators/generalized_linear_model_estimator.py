import itertools
from typing import Any, List, Optional, Union

import pandas as pd
import statsmodels.api as sm

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class GeneralizedLinearModelEstimator(RegressionEstimator):
    """Compute effect of treatment using a generalized linear model such as logistic regression.

    Implementation uses statsmodels.api.GLM.
    Needs an additional parameter, "glm_family" to be specified in method_params. The value of this parameter can be any valid statsmodels.api families object. For example, to use logistic regression, specify "glm_family" as statsmodels.api.families.Binomial().

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
        glm_family: Optional[Any] = None,
        predict_score: bool = True,
        **kwargs,
    ):
        """For a list of args and kwargs, see documentation for
        :class:`~dowhy.causal_estimator.CausalEstimator`.

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
        :param glm_family: statsmodels family for the generalized linear model.
            For example, use statsmodels.api.families.Binomial() for logistic
            regression or statsmodels.api.families.Poisson() for count data.
        :param predict_score: For models that have a binary output, whether
            to output the model's score or the binary output based on the score.
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
            glm_family=glm_family,
            predict_score=predict_score,
            **kwargs,
        )
        self.logger.info("INFO: Using Generalized Linear Model Estimator")
        if glm_family is not None:
            self.family = glm_family
        else:
            raise ValueError(
                "Need to specify the family for the generalized linear model. Provide a 'glm_family' parameter in method_params, such as statsmodels.api.families.Binomial() for logistic regression."
            )
        self.predict_score = predict_score

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
        return super().fit(
            data,
            effect_modifier_names=effect_modifier_names,
        )

    def _build_model(self, data: pd.DataFrame):
        features = self._build_features(data)
        model = sm.GLM(data[self._target_estimand.outcome_variable], features, family=self.family).fit()
        return (features, model)

    def predict_fn(self, data: pd.DataFrame, model, features):
        # Checking if Y is binary
        outcome_values = data[self._target_estimand.outcome_variable[0]].astype(int).unique()
        outcome_is_binary = all([v in [0, 1] for v in outcome_values])

        if outcome_is_binary:
            if self.predict_score:
                return model.predict(features)
            else:
                return (model.predict(features) > 0.5).astype(int)
        else:
            return model.predict(features)

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~" + "Sigmoid("
        var_list = estimand.treatment_variable + estimand.get_adjustment_set()
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = [
                "{0}*{1}".format(x[0], x[1])
                for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)
            ]
            expr += "+" + "+".join(interaction_terms)
        expr += ")"
        return expr
