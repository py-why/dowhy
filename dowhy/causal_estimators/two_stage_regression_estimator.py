import copy
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.utils.api import parse_state


class TwoStageRegressionEstimator(CausalEstimator):
    """Compute treatment effect whenever the effect is fully mediated by
    another variable (front-door) or when there is an instrument available.

    Currently only supports a linear model for the effects.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """

    # First stage statistical model
    DEFAULT_FIRST_STAGE_MODEL = LinearRegressionEstimator
    # Second stage statistical model
    DEFAULT_SECOND_STAGE_MODEL = LinearRegressionEstimator

    def __init__(
        self,
        identified_estimand,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        num_null_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction=CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level=CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates="auto",
        num_quantiles_to_discretize_cont_cols=CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        first_stage_model=None,
        second_stage_model=None,
        **kwargs,
    ):
        """
        :param first_stage_model: First stage estimator to be used. Default is
            linear regression.
        :param second_stage_model: Second stage estimator to be used. Default
            is linear regression.

        """
        # Required to ensure that self.method_params contains all the
        # parameters needed to create an object of this class
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
            first_stage_model=first_stage_model,
            second_stage_model=second_stage_model,
            **kwargs,
        )
        self.logger.info("INFO: Using Two Stage Regression Estimator")
        # Check if the treatment is one-dimensional

        if first_stage_model is not None:
            self.first_stage_model = first_stage_model
        else:
            self.first_stage_model = self.__class__.DEFAULT_FIRST_STAGE_MODEL
            self.logger.warning("First stage model not provided. Defaulting to sklearn.linear_model.LinearRegression.")
        if second_stage_model is not None:
            self.second_stage_model = second_stage_model
        else:
            self.second_stage_model = self.__class__.DEFAULT_SECOND_STAGE_MODEL
            self.logger.warning("Second stage model not provided. Defaulting to backdoor.linear_regression.")

        modified_target_estimand = copy.deepcopy(self._target_estimand)

        self._first_stage_model_obj = self.first_stage_model(
            modified_target_estimand,
            test_significance=self._significance_test,
            evaluate_effect_strength=self._effect_strength_eval,
            confidence_intervals=self._confidence_intervals,
            **self.method_params,
        )

        # Second Stage
        modified_target_estimand = copy.deepcopy(self._target_estimand)

        self._second_stage_model_obj = self.second_stage_model(
            modified_target_estimand,
            test_significance=self._significance_test,
            evaluate_effect_strength=self._effect_strength_eval,
            confidence_intervals=self._confidence_intervals,
            **self.method_params,
        )

        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:
            modified_target_estimand = copy.deepcopy(self._target_estimand)
            self._second_stage_model_nde_obj = self.second_stage_model(
                modified_target_estimand,
                test_significance=self._significance_test,
                evaluate_effect_strength=self._effect_strength_eval,
                confidence_intervals=self._confidence_intervals,
                **self.method_params,
            )

    def fit(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        effect_modifier_names: Optional[List[str]] = None,
        **_,
    ):
        self.set_data(data, treatment_name, outcome_name)
        self.set_effect_modifiers(effect_modifier_names)

        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)

        if self._target_estimand.identifier_method == "frontdoor":
            self.logger.debug("Front-door variable used:" + ",".join(self._target_estimand.get_frontdoor_variables()))
            self._frontdoor_variables_names = self._target_estimand.get_frontdoor_variables()

            if self._frontdoor_variables_names:
                self._frontdoor_variables = self._data[self._frontdoor_variables_names]
            else:
                self._frontdoor_variables = None
                error_msg = "No front-door variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
        elif self._target_estimand.identifier_method == "mediation":
            self.logger.debug("Mediators used:" + ",".join(self._target_estimand.get_mediator_variables()))
            self._mediators_names = self._target_estimand.get_mediator_variables()

            if self._mediators_names:
                self._mediators = self._data[self._mediators_names]
            else:
                self._mediators = None
                error_msg = "No mediator variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
        elif self._target_estimand.identifier_method == "iv":
            self.logger.debug(
                "Instrumental variables used:" + ",".join(self._target_estimand.get_instrumental_variables())
            )
            self._instrumental_variables_names = self._target_estimand.get_instrumental_variables()

            if self._instrumental_variables_names:
                self._instrumental_variables = self._data[self._instrumental_variables_names]
            else:
                self._instrumental_variables = None
                error_msg = "No instrumental variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)

        self._first_stage_model_obj._target_estimand.identifier_method = "backdoor"
        self._first_stage_model_obj._target_estimand.backdoor_variables = (
            self._target_estimand.mediation_first_stage_confounders
        )
        if self._target_estimand.identifier_method == "frontdoor":
            self._first_stage_model_obj._target_estimand.outcome_variable = parse_state(self._frontdoor_variables_names)
        elif self._target_estimand.identifier_method == "mediation":
            self._first_stage_model_obj._target_estimand.outcome_variable = parse_state(self._mediators_names)

        self._first_stage_model_obj.fit(
            data,
            treatment_name,
            parse_state(self._first_stage_model_obj._target_estimand.outcome_variable),
            effect_modifier_names=effect_modifier_names,
        )

        self._second_stage_model_obj._target_estimand.identifier_method = "backdoor"
        self._second_stage_model_obj._target_estimand.backdoor_variables = (
            self._target_estimand.mediation_second_stage_confounders
        )
        if self._target_estimand.identifier_method == "frontdoor":
            self._second_stage_model_obj._target_estimand.treatment_variable = parse_state(
                self._frontdoor_variables_names
            )
        elif self._target_estimand.identifier_method == "mediation":
            self._second_stage_model_obj._target_estimand.treatment_variable = parse_state(self._mediators_names)

        self._second_stage_model_obj.fit(
            data,
            parse_state(self._second_stage_model_obj._target_estimand.treatment_variable),
            parse_state(self._outcome_name),  # to convert it to array before passing to causal estimator)
            effect_modifier_names=effect_modifier_names,
        )

        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:
            self._second_stage_model_nde_obj._target_estimand.identifier_method = "backdoor"
            self._second_stage_model_nde_obj.fit(
                data,
                self._treatment_name,
                parse_state(self._outcome_name),  # to convert it to array before passing to causal estimator)
                effect_modifier_names=effect_modifier_names,
            )

        return self

    def estimate_effect(self, treatment_value: Any = 1, control_value: Any = 0, target_units=None, **_):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value

        estimate_value = None
        # First stage
        first_stage_estimate = self._first_stage_model_obj.estimate_effect(
            control_value=control_value,
            treatment_value=treatment_value,
            target_units=target_units,
        )

        # Second Stage
        second_stage_estimate = self._second_stage_model_obj.estimate_effect(
            control_value=control_value,
            treatment_value=treatment_value,
            target_units=target_units,
        )
        # Combining the two estimates
        natural_indirect_effect = first_stage_estimate.value * second_stage_estimate.value
        # This same estimate is valid for frontdoor as well as mediation (NIE)
        estimate_value = natural_indirect_effect
        self.symbolic_estimator = self.construct_symbolic_estimator(
            first_stage_estimate.realized_estimand_expr,
            second_stage_estimate.realized_estimand_expr,
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
        )
        if self._target_estimand.estimand_type == EstimandType.NONPARAMETRIC_NDE:

            total_effect_estimate = self._second_stage_model_nde_obj.estimate_effect(
                self._data,
                self._treatment_name,
                parse_state(self._outcome_name),
                control_value=control_value,
                treatment_value=treatment_value,
                **self.method_params,
            )
            natural_direct_effect = total_effect_estimate.value - natural_indirect_effect
            estimate_value = natural_direct_effect
            self.symbolic_estimator = self.construct_symbolic_estimator(
                first_stage_estimate.realized_estimand_expr,
                second_stage_estimate.realized_estimand_expr,
                total_effect_estimate.realized_estimand_expr,
                estimand_type=self._target_estimand.estimand_type,
            )
        estimate = CausalEstimate(
            estimate=estimate_value,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def build_first_stage_features(self):
        data_df = self._data
        treatment_vals = data_df[self._treatment_name]
        if len(self._observed_common_causes_names) > 0:
            observed_common_causes_vals = data_df[self._observed_common_causes_names]
            observed_common_causes_vals = pd.get_dummies(observed_common_causes_vals, drop_first=True)
        if self._effect_modifier_names:
            effect_modifiers_vals = data_df[self._effect_modifier_names]
            effect_modifiers_vals = pd.get_dummies(effect_modifiers_vals, drop_first=True)
        if type(treatment_vals) is not np.ndarray:
            treatment_vals = treatment_vals.to_numpy()
        if treatment_vals.shape[0] != data_df.shape[0]:
            raise ValueError("Provided treatment values and dataframe should have the same length.")
        # Bulding the feature matrix
        n_samples = treatment_vals.shape[0]
        self.logger.debug("Number of samples" + str(n_samples) + str(len(self._treatment_name)))
        treatment_2d = treatment_vals.reshape((n_samples, len(self._treatment_name)))
        if len(self._observed_common_causes_names) > 0:
            features = np.concatenate((treatment_2d, observed_common_causes_vals), axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:, i]
                new_features = curr_treatment[:, np.newaxis] * effect_modifiers_vals.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(
            float, copy=False
        )  # converting to float in case of binary treatment and no other variables
        # features = sm.add_constant(features, has_constant='add') # to add an intercept term
        return features

    def construct_symbolic_estimator(
        self, first_stage_symbolic, second_stage_symbolic, total_effect_symbolic=None, estimand_type=None
    ):
        nie_symbolic = "(" + first_stage_symbolic + ")*(" + second_stage_symbolic + ")"
        if estimand_type == EstimandType.NONPARAMETRIC_NIE:
            return nie_symbolic
        elif estimand_type == EstimandType.NONPARAMETRIC_NDE:
            return "(" + total_effect_symbolic + ") - (" + nie_symbolic + ")"
