import numpy as np
import pandas as pd
import itertools
import copy

from dowhy.causal_estimator import CausalEstimator, CausalEstimate
from dowhy.causal_identifier import CausalIdentifier
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.utils.api import parse_state

class TwoStageRegressionEstimator(CausalEstimator):
    """Compute treatment effect whenever the effect is fully mediated by another variable (front-door) or when there is an instrument available.

    Currently only supports a linear model for the effects.
    """
    DEFAULT_FIRST_STAGE_MODEL = LinearRegressionEstimator
    DEFAULT_SECOND_STAGE_MODEL = LinearRegressionEstimator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("INFO: Using Two Stage Regression Estimator")
        # Check if the treatment is one-dimensional
        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)

        if self._target_estimand.identifier_method == "frontdoor":
            self.logger.debug("Front-door variable used:" +
                            ",".join(self._target_estimand.get_frontdoor_variables()))
            self._frontdoor_variables_names = self._target_estimand.get_frontdoor_variables()

            if self._frontdoor_variables_names:
                self._frontdoor_variables = self._data[self._frontdoor_variables_names]
            else:
                self._frontdoor_variables = None
                error_msg = "No front-door variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
        elif self._target_estimand.identifier_method == "mediation":
            self.logger.debug("Mediators used:" +
                            ",".join(self._target_estimand.get_mediator_variables()))
            self._mediators_names = self._target_estimand.get_mediator_variables()

            if self._mediators_names:
               self._mediators = self._data[self._mediators_names]
            else:
                self._mediators = None
                error_msg = "No mediator variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)
        elif self._target_estimand.identifier_method=="iv":
            self.logger.debug("Instrumental variables used:" +
                            ",".join(self._target_estimand.get_instrumental_variables()))
            self._instrumental_variables_names = self._target_estimand.get_instrumental_variables()

            if self._instrumental_variables_names:
                self._instrumental_variables = self._data[self._instrumental_variables_names]
            else:
                self._instrumental_variables = None
                error_msg = "No instrumental variable present. Two stage regression is not applicable"
                self.logger.error(error_msg)

        if 'first_stage_model' in self.method_params:
            self.first_stage_model = self.method_params['first_stage_model']
        else:
            self.first_stage_model = self.__class__.DEFAULT_FIRST_STAGE_MODEL
            self.logger.warning("First stage model not provided. Defaulting to sklearn.linear_model.LinearRegression.")
        if 'second_stage_model' in self.method_params:
            self.second_stage_model = self.method_params['second_stage_model']
        else:
            self.second_stage_model = self.__class__.DEFAULT_SECOND_STAGE_MODEL
            self.logger.warning("Second stage model not provided. Defaulting to backdoor.linear_regression.")

    def _estimate_effect(self):
        #first_stage_features = self.build_first_stage_features()
        #fs_model = self.first_stage_model()
        #if self._target_estimand.identifier_method=="frontdoor":
        #    first_stage_outcome = self._frontdoor_variables
        #elif self._target_estimand.identifier_method=="mediation":
        #    first_stage_outcome = self._mediators
        #fs_model.fit(first_stage_features, self._frontdoor_variables)
        #self.logger.debug("Coefficients of the fitted model: " +
        #                  ",".join(map(str, fs_model.coef_)))
        #residuals = self._frontdoor_variables - fs_model.predict(first_stage_features)
        #self._data["residual"] = residuals
        estimate_value = None
        # First stage
        modified_target_estimand = copy.deepcopy(self._target_estimand)
        modified_target_estimand.identifier_method="backdoor"
        modified_target_estimand.backdoor_variables = self._target_estimand.mediation_first_stage_confounders
        if self._target_estimand.identifier_method=="frontdoor":
            modified_target_estimand.outcome_variable = parse_state(self._frontdoor_variables_names)
        elif self._target_estimand.identifier_method=="mediation":
            modified_target_estimand.outcome_variable = parse_state(self._mediators_names)

        first_stage_estimate = self.first_stage_model(self._data,
                 modified_target_estimand,
                 self._treatment_name,
                 parse_state(modified_target_estimand.outcome_variable),
                 control_value=self._control_value,
                 treatment_value=self._treatment_value,
                 test_significance=self._significance_test,
                 evaluate_effect_strength=self._effect_strength_eval,
                 confidence_intervals = self._confidence_intervals,
                 target_units=self._target_units,
                 effect_modifiers=self._effect_modifier_names,
                 params=self.method_params)._estimate_effect()

        # Second Stage
        modified_target_estimand = copy.deepcopy(self._target_estimand)
        modified_target_estimand.identifier_method="backdoor"
        modified_target_estimand.backdoor_variables = self._target_estimand.mediation_second_stage_confounders
        if self._target_estimand.identifier_method=="frontdoor":
            modified_target_estimand.treatment_variable = parse_state(self._frontdoor_variables_names)
        elif self._target_estimand.identifier_method=="mediation":
            modified_target_estimand.treatment_variable = parse_state(self._mediators_names)

        second_stage_estimate = self.second_stage_model(self._data,
                 modified_target_estimand,
                 parse_state(modified_target_estimand.treatment_variable),
                 parse_state(self._outcome_name), # to convert it to array before passing to causal estimator
                 control_value=self._control_value,
                 treatment_value=self._treatment_value,
                 test_significance=self._significance_test,
                 evaluate_effect_strength=self._effect_strength_eval,
                 confidence_intervals = self._confidence_intervals,
                 target_units=self._target_units,
                 effect_modifiers=self._effect_modifier_names,
                 params=self.method_params)._estimate_effect()
        # Combining the two estimates
        natural_indirect_effect = first_stage_estimate.value * second_stage_estimate.value
        # This same estimate is valid for frontdoor as well as mediation (NIE)
        estimate_value = natural_indirect_effect
        self.symbolic_estimator = self.construct_symbolic_estimator(
                first_stage_estimate.realized_estimand_expr,
                second_stage_estimate.realized_estimand_expr,
                estimand_type=CausalIdentifier.NONPARAMETRIC_NIE)
        if self._target_estimand.estimand_type == CausalIdentifier.NONPARAMETRIC_NDE:
            # Total  effect of treatment
            modified_target_estimand = copy.deepcopy(self._target_estimand)
            modified_target_estimand.identifier_method="backdoor"

            total_effect_estimate = self.second_stage_model(self._data,
                     modified_target_estimand,
                     self._treatment_name,
                     parse_state(self._outcome_name),
                     control_value=self._control_value,
                     treatment_value=self._treatment_value,
                     test_significance=self._significance_test,
                     evaluate_effect_strength=self._effect_strength_eval,
                     confidence_intervals = self._confidence_intervals,
                     target_units=self._target_units,
                     effect_modifiers=self._effect_modifier_names,
                     params=self.method_params)._estimate_effect()
            natural_direct_effect = total_effect_estimate.value - natural_indirect_effect
            estimate_value = natural_direct_effect
            self.symbolic_estimator = self.construct_symbolic_estimator(
                    first_stage_estimate.realized_estimand_expr,
                    second_stage_estimate.realized_estimand_expr,
                    total_effect_estimate.realized_estimand_expr,
                    estimand_type=self._target_estimand.estimand_type)
        return CausalEstimate(estimate=estimate_value,
                              target_estimand=self._target_estimand,
                              realized_estimand_expr=self.symbolic_estimator)

    def build_first_stage_features(self):
        data_df = self._data
        treatment_vals = data_df[self._treatment_name]
        if len(self._observed_common_causes_names)>0:
            observed_common_causes_vals = data_df[self._observed_common_causes_names]
            observed_common_causes_vals = pd.get_dummies(observed_common_causes_vals, drop_first=True)
        if self._effect_modifier_names:
            effect_modifiers_vals =  data_df[self._effect_modifier_names]
            effect_modifiers_vals = pd.get_dummies(effect_modifiers_vals, drop_first=True)
        if type(treatment_vals) is not np.ndarray:
            treatment_vals = treatment_vals.to_numpy()
        if treatment_vals.shape[0] != data_df.shape[0]:
            raise ValueError("Provided treatment values and dataframe should have the same length.")
        # Bulding the feature matrix
        n_samples = treatment_vals.shape[0]
        self.logger.debug("Number of samples" +str(n_samples) + str(len(self._treatment_name)))
        treatment_2d = treatment_vals.reshape((n_samples,len(self._treatment_name)))
        if len(self._observed_common_causes_names)>0:
            features = np.concatenate((treatment_2d, observed_common_causes_vals),
                                  axis=1)
        else:
            features = treatment_2d
        if self._effect_modifier_names:
            for i in range(treatment_2d.shape[1]):
                curr_treatment = treatment_2d[:,i]
                new_features = curr_treatment[:, np.newaxis] * effect_modifiers_vals.to_numpy()
                features = np.concatenate((features, new_features), axis=1)
        features = features.astype(float, copy=False) # converting to float in case of binary treatment and no other variables
        #features = sm.add_constant(features, has_constant='add') # to add an intercept term
        return features

    def construct_symbolic_estimator(self, first_stage_symbolic,
            second_stage_symbolic, total_effect_symbolic=None, estimand_type=None):
        nie_symbolic = "(" + first_stage_symbolic + ")*(" + second_stage_symbolic + ")"
        if estimand_type == CausalIdentifier.NONPARAMETRIC_NIE:
            return nie_symbolic
        elif estimand_type == CausalIdentifier.NONPARAMETRIC_NDE:
            return "(" + total_effect_symbolic + ") - (" + nie_symbolic + ")"

