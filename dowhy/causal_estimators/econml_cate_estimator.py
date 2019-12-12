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
            raise ImportError('{}.{} is not an existing causal estimator.'.format(module_name, class_name))
        return estimator_class


    def _estimate_effect(self):
        n_samples = self._treatment.shape[0]
        X = None  # Effect modifiers
        W = None  # common causes/ confounders
        Y = np.ndarray(shape=(n_samples,1), buffer=np.array(self._outcome))
        T = np.ndarray(shape=(n_samples,1), buffer=np.array(self._treatment))
        if self._effect_modifier_names:
            X = np.ndarray(shape=(n_samples,1), buffer=np.array(self._effect_modifiers))
        if self._observed_common_causes_names:
            W = np.ndarray(shape=(n_samples,1), buffer=np.array(self._observed_common_causes))
        if self._instrumental_variable_names:
            Z = np.ndarray(shape=(n_samples, 1), buffer=np.array(self._instrumental_variables))

        # Calling the econml estimator's fit method
        if self.identifier_method == "backdoor":
            self.estimator.fit(Y, T, X, W, **self.method_params["fit_params"])
        else:
            self.estimator.fit(Y, T, X, Z, **self.method_params["fit_params"])

        X_test = X
        n_target_units = n_samples
        if X is not None:
            filtered_rows = self._data.where(self._target_units)
            boolean_criterion = np.array(filtered_rows.notnull().iloc[:,0])
            X_test = X[boolean_criterion]
            n_target_units = X_test.shape[0]
        T0_test = np.repeat([[self._control_value]], n_target_units, axis=0)
        T1_test = np.repeat([[self._treatment_value]], n_target_units, axis=0)
        est = self.estimator.effect(X_test, T0 = T0_test, T1 = T1_test)
        ate = np.mean(est)

        est_interval = None
        if self._confidence_intervals:
            est_interval = self.estimator.effect_interval(X_test, T0 = T0_test, T1 = T1_test)
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  cate_estimates=est,
                                  effect_intervals=est_interval)
        return estimate


    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
