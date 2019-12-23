import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator
from importlib import import_module
import econml

class EconmlCateEstimator(CausalEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identifier_method = self._target_estimand.identifier_method
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))

        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        if self._observed_common_causes_names:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes= None
            error_msg ="No common causes/confounders present."
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Instrumental variables names, if present
        self._instrumental_variable_names = self._target_estimand.instrumental_variables
        if self._instrumental_variable_names:
            self._instrumental_variables = self._data[self._instrumental_variable_names]
            self._instrumental_variables = pd.get_dummies(self._instrumental_variables, drop_first=True)
        else:
            self._instrumental_variables = None

        estimator_class = self._get_econml_class_object(self._econml_methodname)
        self.estimator = estimator_class(**self.method_params["init_params"])
        self.logger.info("INFO: Using EconML Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _get_econml_class_object(self, module_method_name, *args, **kwargs):
        # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
        try:
            (module_name, _, class_name) = module_method_name.rpartition(".")
            estimator_module = import_module(module_name)
            estimator_class = getattr(estimator_module, class_name)

        except (AttributeError, AssertionError, ImportError):
            raise ImportError('Error loading {}.{}. Double-check the method name and ensure that all econml dependencies are installed.'.format(module_name, class_name))
        return estimator_class


    def _estimate_effect(self):
        n_samples = self._treatment.shape[0]
        X = None  # Effect modifiers
        W = None  # common causes/ confounders
        Y = np.reshape(np.array(self._outcome),(n_samples, 1))
        T = np.reshape(np.array(self._treatment), (n_samples, len(self._treatment_name)))
        if self._effect_modifier_names:
            X = np.reshape(np.array(self._effect_modifiers), (n_samples, len(self._effect_modifier_names)))
        if self._observed_common_causes_names:
            W = np.reshape(np.array(self._observed_common_causes), (n_samples,len(self._observed_common_causes_names)))
        if self._instrumental_variable_names:
            Z = np.reshape(np.array(self._instrumental_variables), (n_samples, len(self._instrumental_variable_names)))

        # Calling the econml estimator's fit method
        if self.identifier_method == "backdoor":
            self.estimator.fit(Y, T, X, W, **self.method_params["fit_params"])
        else:
            self.estimator.fit(Y, T, X, Z, **self.method_params["fit_params"])

        X_test = X
        n_target_units = n_samples
        if X is not None:
            if type(self._target_units) is pd.DataFrame:
                X_test = self._target_units
            elif callable(self._target_units):
                filtered_rows = self._data.where(self._target_units)
                boolean_criterion = np.array(filtered_rows.notnull().iloc[:,0])
                X_test = X[boolean_criterion]
            n_target_units = X_test.shape[0]

        # Changing shape to a list for a singleton value
        if type(self._control_value) is not list:
            self._control_value = [self._control_value]
        if type(self._treatment_value) is not list:
            self._treatment_value = [self._treatment_value]
        T0_test = np.repeat([self._control_value], n_target_units, axis=0)
        T1_test = np.repeat([self._treatment_value], n_target_units, axis=0)
        est = self.estimator.effect(X_test, T0 = T0_test, T1 = T1_test)
        ate = np.mean(est)

        est_interval = None
        if self._confidence_intervals:
            est_interval = self.estimator.effect_interval(X_test, T0 = T0_test, T1 = T1_test)
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  cate_estimates=est,
                                  effect_intervals=est_interval,
                                  _estimator_object = self.estimator)
        return estimate

    def _do(self, x):
        raise NotImplementedError

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
