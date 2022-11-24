import inspect
from importlib import import_module
from typing import Callable

import numpy as np
import pandas as pd
from numpy.distutils.misc_util import is_sequence

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.utils.api import parse_state


class Econml(CausalEstimator):
    """Wrapper class for estimators from the EconML library.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below. For init and fit
    parameters of each estimator, refer to the EconML docs.

    """

    def __init__(self, *args, econml_methodname, **kwargs):
        """
        :param econml_methodname: Fully qualified name of econml estimator
            class. For example, 'econml.dml.DML'
        """
        # Required to ensure that self.method_params contains all the
        # parameters to create an object of this class
        args_dict = {k: v for k, v in locals().items() if k not in type(self)._STD_INIT_ARGS}
        args_dict.update(kwargs)
        super().__init__(*args, **args_dict)
        self._econml_methodname = econml_methodname
        self.logger.info("INFO: Using EconML Estimator")
        self.identifier_method = self._target_estimand.identifier_method
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables().copy()
        # For metalearners only--issue a warning if w contains variables not in x
        (module_name, _, class_name) = self._econml_methodname.rpartition(".")
        if module_name.endswith("metalearners"):
            effect_modifier_names = []
            if self._effect_modifier_names is not None:
                effect_modifier_names = self._effect_modifier_names.copy()
            w_diff_x = [w for w in self._observed_common_causes_names if w not in effect_modifier_names]
            if len(w_diff_x) > 0:
                self.logger.warn(
                    "Concatenating common_causes and effect_modifiers and providing a single list of variables to metalearner estimator method, "
                    + class_name
                    + ". EconML metalearners accept a single X argument."
                )
                effect_modifier_names.extend(w_diff_x)
                # Override the effect_modifiers set in CausalEstimator.__init__()
                # Also only update self._effect_modifiers, and create a copy of self._effect_modifier_names
                # the latter can be used by other estimator methods later
                self._effect_modifiers = self._data[effect_modifier_names]
                self._effect_modifiers = pd.get_dummies(self._effect_modifiers, drop_first=True)
                self._effect_modifier_names = effect_modifier_names
            self.logger.debug("Effect modifiers: " + ",".join(effect_modifier_names))
        if self._observed_common_causes_names:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = None
        self.logger.debug("Back-door variables used:" + ",".join(self._observed_common_causes_names))
        # Instrumental variables names, if present
        # choosing the instrumental variable to use
        if getattr(self, "iv_instrument_name", None) is None:
            self.estimating_instrument_names = self._target_estimand.instrumental_variables
        else:
            self.estimating_instrument_names = parse_state(self.iv_instrument_name)
        if self.estimating_instrument_names:
            self._estimating_instruments = self._data[self.estimating_instrument_names]
            self._estimating_instruments = pd.get_dummies(self._estimating_instruments, drop_first=True)
        else:
            self._estimating_instruments = None
        self.estimator = None
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _get_econml_class_object(self, module_method_name, *args, **kwargs):
        # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
        try:
            (module_name, _, class_name) = module_method_name.rpartition(".")
            estimator_module = import_module(module_name)
            estimator_class = getattr(estimator_module, class_name)

        except (AttributeError, AssertionError, ImportError):
            raise ImportError(
                "Error loading {}.{}. Double-check the method name and ensure that all econml dependencies are installed.".format(
                    module_name, class_name
                )
            )
        return estimator_class

    def _estimate_effect(self):
        n_samples = self._treatment.shape[0]
        X = None  # Effect modifiers
        W = None  # common causes/ confounders
        Z = None  # Instruments
        Y = self._outcome
        T = self._treatment
        if self._effect_modifiers is not None:
            X = self._effect_modifiers
        if self._observed_common_causes_names:
            W = self._observed_common_causes
        if self.estimating_instrument_names:
            Z = self._estimating_instruments
        named_data_args = {"Y": Y, "T": T, "X": X, "W": W, "Z": Z}

        if self.estimator is None:
            estimator_class = self._get_econml_class_object(self._econml_methodname)
            self.estimator = estimator_class(**self.method_params["init_params"])
            # Calling the econml estimator's fit method
            estimator_argspec = inspect.getfullargspec(inspect.unwrap(self.estimator.fit))
            # As of v0.9, econml has some kewyord only arguments
            estimator_named_args = estimator_argspec.args + estimator_argspec.kwonlyargs
            estimator_data_args = {
                arg: named_data_args[arg] for arg in named_data_args.keys() if arg in estimator_named_args
            }
            if self.method_params["fit_params"] is not False:
                self.estimator.fit(**estimator_data_args, **self.method_params["fit_params"])

        X_test = X
        if X is not None:
            if type(self._target_units) is pd.DataFrame:
                X_test = self._target_units
            elif callable(self._target_units):
                filtered_rows = self._data.where(self._target_units)
                boolean_criterion = np.array(filtered_rows.notnull().iloc[:, 0])
                X_test = X[boolean_criterion]
        # Changing shape to a list for a singleton value
        self._treatment_value = parse_state(self._treatment_value)

        est = self.effect(X_test)
        ate = np.mean(est, axis=0)  # one value per treatment value

        if len(ate) == 1:
            ate = ate[0]

        if self._confidence_intervals:
            self.effect_intervals = self.effect_interval(X_test)
        else:
            self.effect_intervals = None

        estimate = CausalEstimate(
            estimate=ate,
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            cate_estimates=est,
            effect_intervals=self.effect_intervals,
            _estimator_object=self.estimator,
        )
        return estimate

    def _estimate_confidence_intervals(self, confidence_level=None, method=None):
        """Returns None if the confidence interval has not been calculated."""
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

    def shap_values(self, df: pd.DataFrame, *args, **kwargs):
        return self.estimator.shap_values(df[self._effect_modifier_names].values, *args, **kwargs)

    def apply_multitreatment(self, df: pd.DataFrame, fun: Callable, *args, **kwargs):
        ests = []
        assert not isinstance(self._treatment_value, str)
        assert is_sequence(self._treatment_value)

        if df is None:
            filtered_df = None
        else:
            filtered_df = df[self._effect_modifier_names].values

        for tv in self._treatment_value:
            ests.append(
                fun(
                    filtered_df,
                    T0=self._control_value,
                    T1=tv,
                    *args,
                    **kwargs,
                )
            )
        est = np.stack(ests, axis=1)
        return est

    def effect(self, df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """
        Pointwise estimated treatment effect,
        output shape n_units x n_treatment_values (not counting control)
        :param df: Features of the units to evaluate
        :param args: passed through to the underlying estimator
        :param kwargs: passed through to the underlying estimator
        """

        def effect_fun(filtered_df, T0, T1, *args, **kwargs):
            return self.estimator.effect(filtered_df, T0=T0, T1=T1, *args, **kwargs)

        return self.apply_multitreatment(df, effect_fun, *args, **kwargs)

    def effect_interval(self, df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """
        Pointwise confidence intervals for the estimated treatment effect
        :param df: Features of the units to evaluate
        :param args: passed through to the underlying estimator
        :param kwargs: passed through to the underlying estimator
        """

        def effect_interval_fun(filtered_df, T0, T1, *args, **kwargs):
            return self.estimator.effect_interval(
                filtered_df, T0=T0, T1=T1, alpha=1 - self.confidence_level, *args, **kwargs
            )

        return self.apply_multitreatment(df, effect_interval_fun, *args, **kwargs)

    def effect_inference(self, df: pd.DataFrame, *args, **kwargs):
        """
        Inference (uncertainty) results produced by the underlying EconML estimator
        :param df: Features of the units to evaluate
        :param args: passed through to the underlying estimator
        :param kwargs: passed through to the underlying estimator
        """

        def effect_inference_fun(filtered_df, T0, T1, *args, **kwargs):
            return self.estimator.effect_inference(filtered_df, T0=T0, T1=T1, *args, **kwargs)

        return self.apply_multitreatment(df, effect_inference_fun, *args, **kwargs)

    def effect_tt(self, df: pd.DataFrame, *args, **kwargs):
        """
        Effect of the actual treatment that was applied to each unit
        ("effect of Treatment on the Treated")
        :param df: Features of the units to evaluate
        :param args: passed through to estimator.effect()
        :param kwargs: passed through to estimator.effect()
        """

        eff = self.effect(df, *args, **kwargs).reshape((len(df), len(self._treatment_value)))

        out = np.zeros(len(df))
        treatment_value = parse_state(self._treatment_value)
        treatment_name = parse_state(self._treatment_name)[0]

        eff = np.reshape(eff, (len(df), len(treatment_value)))

        # For each unit, return the estimated effect of the treatment value
        # that was actually applied to the unit
        for c, col in enumerate(treatment_value):
            out[df[treatment_name] == col] = eff[df[treatment_name] == col, c]
        return pd.Series(data=out, index=df.index)
