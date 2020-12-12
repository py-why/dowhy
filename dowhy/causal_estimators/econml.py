import inspect
import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator
from importlib import import_module
import econml

class Econml(CausalEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("INFO: Using EconML Estimator")
        self.identifier_method = self._target_estimand.identifier_method
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables().copy()
        # Checking if effect modifiers are a subset of common causes
        x_subsetof_w = True
        unique_effect_modifier_names = []
        for em_name in self._effect_modifier_names:
            if em_name not in self._observed_common_causes_names:
                x_subsetof_w = False
                unique_effect_modifier_names.append(em_name)
        if not x_subsetof_w:
            self.logger.warn("Effect modifiers are not a subset of common causes. For efficiency in estimation, EconML will consider all effect modifiers as common causes too.")
            self._observed_common_causes_names.extend(unique_effect_modifier_names)

        # For metalearners only--issue a warning if w contains variables not in x
        (module_name, _, class_name) = self._econml_methodname.rpartition(".")
        if module_name.endswith("metalearners"):
            effect_modifier_names = []
            if self._effect_modifier_names is not None:
                effect_modifier_names = self._effect_modifier_names.copy()
            w_diff_x = [w for w in self._observed_common_causes_names if w not in effect_modifier_names]
            if len(w_diff_x) >0:
                self.logger.warn("Concatenating common_causes and effect_modifiers and providing a single list of variables to metalearner estimator method, " + class_name + ". EconML metalearners accept a single X argument.")
                effect_modifier_names.extend(w_diff_x)
                # Override the effect_modifiers set in CausalEstimator.__init__()
                # Also only update self._effect_modifiers, and create a copy of self._effect_modifier_names
                # the latter can be used by other estimator methods later
                self._effect_modifiers = self._data[effect_modifier_names]
                self._effect_modifiers = pd.get_dummies(self._effect_modifiers, drop_first=True)
                self._effect_modifier_names = effect_modifier_names
            self.logger.debug("Effect modifiers: " +
                          ",".join(effect_modifier_names))
        if self._observed_common_causes_names:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = None
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._observed_common_causes_names))
        # Instrumental variables names, if present
        self._instrumental_variable_names = self._target_estimand.instrumental_variables
        if self._instrumental_variable_names:
            self._instrumental_variables = self._data[self._instrumental_variable_names]
            self._instrumental_variables = pd.get_dummies(self._instrumental_variables, drop_first=True)
        else:
            self._instrumental_variables = None

        estimator_class = self._get_econml_class_object(self._econml_methodname)
        self.estimator = estimator_class(**self.method_params["init_params"])
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
        Z = None  # Instruments
        Y = np.array(self._outcome)
        T = np.array(self._treatment)
        if self._effect_modifiers is not None:
            X = np.reshape(np.array(self._effect_modifiers), (n_samples, self._effect_modifiers.shape[1]))
        if self._observed_common_causes_names:
            W = np.reshape(np.array(self._observed_common_causes), (n_samples, self._observed_common_causes.shape[1]))
        if self._instrumental_variable_names:
            Z = np.array(self._instrumental_variables)
        named_data_args = {'Y': Y, 'T': T, 'X': X, 'W': W, 'Z': Z}

        # Calling the econml estimator's fit method
        estimator_named_args = inspect.getfullargspec(
            inspect.unwrap(self.estimator.fit)
            )[0]
        estimator_data_args = {
            arg: named_data_args[arg] for arg in named_data_args.keys() if arg in estimator_named_args
            }
        self.estimator.fit(**estimator_data_args, **self.method_params["fit_params"])

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
        est = self.estimator.effect(X_test, T0=T0_test, T1=T1_test)
        ate = np.mean(est)

        self.effect_intervals = None
        if self._confidence_intervals:
            self.effect_intervals = self.estimator.effect_interval(
                    X_test, T0=T0_test, T1=T1_test,
                    alpha=1-self.confidence_level)
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  cate_estimates=est,
                                  effect_intervals=self.effect_intervals,
                                  _estimator_object=self.estimator)
        return estimate

    def _estimate_confidence_intervals(self, confidence_level=None, method=None):
        """ Returns None if the confidence interval has not been calculated.
        """
        return self.effect_intervals

    def _do(self, x):
        raise NotImplementedError

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        (module_name, _, class_name) = self._econml_methodname.rpartition(".")
        if module_name.endswith("metalearners"):
            var_list = estimand.treatment_variable + self._effect_modifier_names
            expr += "+".join(var_list)
        else:
            var_list = estimand.treatment_variable + self._observed_common_causes_names
            expr += "+".join(var_list)
            expr += " | " + ",".join(self._effect_modifier_names)
        return expr
