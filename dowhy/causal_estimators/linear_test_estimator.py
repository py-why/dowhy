import itertools
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dowhy.causal_estimator import CausalEstimate, CausalEstimator, IdentifiedEstimand
from dowhy.causal_estimators.regression_estimator import RegressionEstimator


class LinearTestEstimator(RegressionEstimator):
    """
    Compute effect of treatment using a linear regression model.
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

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation.
        """
        self.reset_encoders()
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

        # 모델 학습 (sklearn 사용)
        features = self._build_features(data)
        outcome = data[self._target_estimand.outcome_variable].to_numpy()
        self.model.fit(features, outcome)

        # sklearn 모델의 계수를 로깅
        self.logger.debug("Coefficients of the fitted model: " + ",".join(map(str, self.model.coef_)))
        self.logger.debug("Intercept of the fitted model: " + str(self.model.intercept_))

        return self

    def _build_model(self, data: pd.DataFrame):
        """
        Builds the scikit-learn LinearRegression model.
        This method is required by the parent class but the logic is now in `fit`.
        """
        return (None, self.model)

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

        # All treatments are set to the same constant value
        effect_estimate = self._do(treatment_value, data) - self._do(control_value, data)
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                data, self._estimate_effect_fn, effect_modifier_names=self._effect_modifier_names
            )

        # `sklearn` 모델은 절편을 `intercept_` 속성에 저장합니다.
        intercept_parameter = self.model.intercept_

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
