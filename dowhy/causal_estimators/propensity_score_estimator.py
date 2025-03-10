from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.exceptions import NotFittedError

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class PropensityScoreEstimator(CausalEstimator):
    """
    Base class for estimators that estimate effects based on propensity of
    treatment assignment.

    Supports additional parameters as listed below.

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
        propensity_score_model: Optional[Any] = None,
        propensity_score_column: str = "propensity_score",
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
        :param propensity_score_model: Model used to compute propensity score.
            Can be any classification model that supports fit() and
            predict_proba() methods. If None, LogisticRegression is used.
        :param propensity_score_column: Column name that stores the
            propensity score. Default='propensity_score'
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
            propensity_score_model=propensity_score_model,
            propensity_score_column=propensity_score_column,
            **kwargs,
        )

        # Enable the user to pass params for a custom propensity model
        self.propensity_score_model = propensity_score_model
        self.propensity_score_column = propensity_score_column

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        self.reset_encoders()  # Forget any existing encoders
        self._set_effect_modifiers(data, effect_modifier_names)

        self.logger.debug("Adjustment set variables used:" + ",".join(self._target_estimand.get_adjustment_set()))
        self._observed_common_causes_names = self._target_estimand.get_adjustment_set()

        if self._observed_common_causes_names:
            self._observed_common_causes = data[self._observed_common_causes_names]
            # Convert the categorical variables into dummy/indicator variables
            # Basically, this gives a one hot encoding for each category
            # The first category is taken to be the base line.
            self._observed_common_causes = self._encode(self._observed_common_causes, "observed_common_causes")

        else:
            self._observed_common_causes = None
            error_msg = "No common causes/confounders present. Propensity score based methods are not applicable"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Check if the treatment is one-dimensional
        if len(self._target_estimand.treatment_variable) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        # Checking if the treatment is binary
        treatment_values = data[self._target_estimand.treatment_variable[0]].astype(int).unique()
        if any([v not in [0, 1] for v in treatment_values]):
            error_msg = "Propensity score methods are applicable only for binary treatments"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        if self.propensity_score_column not in data:
            if self.propensity_score_model is None:
                self.propensity_score_model = linear_model.LogisticRegression()
            treatment_reshaped = np.ravel(data[self._target_estimand.treatment_variable])
            self.propensity_score_model.fit(self._observed_common_causes, treatment_reshaped)

        return self

    def estimate_propensity_score_column(self, data):
        try:
            data[self.propensity_score_column] = self.propensity_score_model.predict_proba(
                self._observed_common_causes
            )[:, 1]
        except NotFittedError:
            raise NotFittedError("Please fit the propensity score model before calling predict_proba")

    def construct_symbolic_estimator(self, estimand):
        """
        A symbolic string that conveys what each estimator does.
        For instance, linear regression is expressed as
        y ~ bx + e
        """
        raise NotImplementedError
