import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression

from dowhy.causal_estimator import CausalEstimate, CausalEstimator, IdentifiedEstimand
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from typing import Any, List, Optional, Union


class LinearTestEstimator(RegressionEstimator):
    """
    Compute effect of treatment using a linear regression model.
    This is a test implementation to verify the CausalEstimator integration.
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
        self.logger.info("INFO: Using LinearTestEstimator")
        self.model = LinearRegression()

    def _build_model(self, data: pd.DataFrame):
        """
        Builds the scikit-learn LinearRegression model.
        """
        features = self._build_features(data)
        outcome = data[self._target_estimand.outcome_variable].to_numpy()
        self.model.fit(features, outcome)
        return (features, self.model)

    def predict_fn(self, data: pd.DataFrame, model, features):
        """
        Predicts outcomes using the fitted scikit-learn model.
        """
        predictions = model.predict(features)
        return predictions

    def construct_symbolic_estimator(self, estimand):
        """
        Constructs a symbolic expression for the estimator.
        """
        expr = "Y ~ " + " + ".join(estimand.treatment_variable + estimand.get_adjustment_set())
        if self._effect_modifier_names:
            interaction_terms = [
                f"{x[0]}*{x[1]}" for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)
            ]
            expr += " + " + " + ".join(interaction_terms)
        return expr