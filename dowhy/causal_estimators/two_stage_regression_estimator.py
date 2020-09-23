import numpy as np
import pandas as pd
import itertools
import copy
from sklearn import linear_model

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator

class TwoStageRegressionEstimator(CausalEstimator):
    """Compute treatment effect whenever the effect is fully mediated by another variable (front-door) or when there is an instrument available.

    Currently only supports a linear model for the effects.
    """
    DEFAULT_FIRST_STAGE_MODEL = linear_model.LinearRegression
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

        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()

    def _estimate_effect(self):
        first_stage_features = self.build_first_stage_features()
        fs_model = self.first_stage_model()
        fs_model.fit(first_stage_features, self._frontdoor_variables)
        self.logger.debug("Coefficients of the fitted model: " +
                          ",".join(map(str, fs_model.coef_)))
        residuals = self._frontdoor_variables - fs_model.predict(first_stage_features)
        self._data["residual"] = residuals
        modified_target_estimand = copy.deepcopy(self._target_estimand)
        modified_target_estimand.treatment_variable = ["residual",]
        estimate = self.second_stage_model(self._data, 
                 modified_target_estimand,  ["residual",], self._outcome_name,
                 control_value=self._control_value, 
                 treatment_value=self._treatment_value,
                 test_significance=self._significance_test, 
                 evaluate_effect_strength=self._effect_strength_eval,
                 confidence_intervals = self._confidence_intervals,
                 target_units=self._target_units, 
                 effect_modifiers=self._effect_modifier_names,
                 params=self.method_params)._estimate_effect()
        estimate.value = estimate.value*fs_model.coef_[0]
        return estimate
    
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
