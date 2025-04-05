import inspect
from importlib import import_module
from typing import Any, Callable, List, Optional, Protocol, Union
from warnings import warn

import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_identifier import IdentifiedEstimand
from dowhy.utils.api import parse_state


class _EconmlEstimator(Protocol):
    def fit(self, *args, **kwargs): ...

    def effect(self, *args, **kwargs): ...

    def effect_interval(self, *args, **kwargs): ...

    def effect_inference(self, *args, **kwargs): ...

    def shap_values(self, *args, **kwargs): ...


class Econml(CausalEstimator):
    """Wrapper class for estimators from the EconML library.

    Supports additional parameters as listed below. For init and fit
    parameters of each estimator, refer to the EconML docs.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        econml_estimator: Union[_EconmlEstimator, str],
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
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param econml_estimator: Instance of an econml estimator class.
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
            econml_estimator=econml_estimator,
            **kwargs,
        )

        if isinstance(econml_estimator, str):
            warn(
                "Using a string to specify the value for econml_estimator is now deprecated, please provide an instance of a econml object",
                DeprecationWarning,
                stacklevel=2,
            )
            estimator_class = self._get_econml_class_object(econml_estimator)
            self.estimator = estimator_class(**kwargs["init_params"])
        else:
            self.estimator = econml_estimator

        self.logger.info("INFO: Using EconML Estimator")
        self.identifier_method = self._target_estimand.identifier_method

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        self.reset_encoders()  # Forget any existing encoders
        self._set_effect_modifiers(data, effect_modifier_names)
        # Save parameters for later refutter fitting
        self._econml_fit_params = kwargs
        self._fit_params = kwargs

        self._observed_common_causes_names = self._target_estimand.get_adjustment_set().copy()

        # Enforcing this ordering is necessary to feed through the propensity values from dataset
        self._observed_common_causes_names = [
            c for c in self._observed_common_causes_names if "propensity" not in c
        ] + sorted([c for c in self._observed_common_causes_names if "propensity" in c])

        # For metalearners only--issue a warning if w contains variables not in x
        if self.estimator.__module__.endswith("metalearners"):
            effect_modifier_names = []
            if self._effect_modifier_names is not None:
                effect_modifier_names = self._effect_modifier_names.copy()
            w_diff_x = [w for w in self._observed_common_causes_names if w not in effect_modifier_names]
            if len(w_diff_x) > 0:
                self.logger.warning(
                    "Concatenating common_causes and effect_modifiers and providing a single list of variables to metalearner estimator method, "
                    + self.estimator.__class__.__name__
                    + ". EconML metalearners accept a single X argument."
                )
                effect_modifier_names.extend(w_diff_x)
                # Override the effect_modifiers set in CausalEstimator.__init__()
                # Also only update self._effect_modifiers, and create a copy of self._effect_modifier_names
                # the latter can be used by other estimator methods later
                self._effect_modifiers = data[effect_modifier_names]
                self._effect_modifiers = self._encode(self._effect_modifiers, "effect_modifiers")
                self._effect_modifier_names = effect_modifier_names
            self.logger.debug("Effect modifiers: " + ",".join(effect_modifier_names))
        if self._observed_common_causes_names:
            self._observed_common_causes = data[self._observed_common_causes_names]
            self._observed_common_causes = self._encode(self._observed_common_causes, "observed_common_causes")
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
            self._estimating_instruments = data[self.estimating_instrument_names]
            self._estimating_instruments = self._encode(self._estimating_instruments, "estimating_instruments")
        else:
            self._estimating_instruments = None

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        X = None
        W = None  # common causes/ confounders
        Z = None  # Instruments
        Y = data[self._target_estimand.outcome_variable[0]]
        T = data[self._target_estimand.treatment_variable]
        if self._effect_modifiers is not None and len(self._effect_modifiers) > 0:
            X = self._effect_modifiers
        if self._observed_common_causes_names:
            W = self._observed_common_causes
        if self.estimating_instrument_names:
            Z = self._estimating_instruments
        named_data_args = {"Y": Y, "T": T, "X": X, "W": W, "Z": Z}
        # Calling the econml estimator's fit method
        estimator_argspec = inspect.getfullargspec(inspect.unwrap(self.estimator.fit))
        # As of v0.9, econml has some kewyord only arguments
        estimator_named_args = estimator_argspec.args + estimator_argspec.kwonlyargs
        estimator_data_args = {
            arg: named_data_args[arg] for arg in named_data_args.keys() if arg in estimator_named_args
        }
        self.estimator.fit(**estimator_data_args, **kwargs)

        return self

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

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        **_,
    ):
        """
        data: dataframe containing the data on which treatment effect is to be estimated.
        treatment_value: value of the treatment variable for which the effect is to be estimated. It can be (optionally) a sequence for different values of the treatment variable.
        control_value: value of the treatment variable that denotes its absence (usually 0)
        target_units: The units for which the treatment effect should be estimated.
                     It can be a DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
                     It can also be a lambda function that can be used as an index for the data (pandas DataFrame) to select the required rows.
        """
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value

        X = None  # Effect modifiers
        if self._effect_modifiers is not None and len(self._effect_modifiers) > 0:
            X = self._effect_modifiers

        X_test = X
        if X is not None:
            if type(target_units) is pd.DataFrame:
                X_test = target_units
            elif callable(target_units):
                filtered_rows = data.where(target_units)
                boolean_criterion = np.array(filtered_rows.notnull().iloc[:, 0])
                X_test = X[boolean_criterion]
        # Changing shape to a list for a singleton value
        # Note that self._control_value is assumed to be a singleton value
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
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=ate,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            cate_estimates=est,
            effect_intervals=self.effect_intervals,
            _estimator_object=self.estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def _estimate_confidence_intervals(self, confidence_level=None, method=None):
        """Returns None if the confidence interval has not been calculated."""
        return self.effect_intervals

    def _do(self, x, data_df=None):
        raise NotImplementedError

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        if self.estimator.__module__.endswith("metalearners"):
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

        if df is None:
            filtered_df = None
        else:
            filtered_df = df.values
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

        Xdf = df[self._effect_modifier_names] if df is not None else df
        return self.apply_multitreatment(Xdf, effect_fun, *args, **kwargs)

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

        Xdf = df[self._effect_modifier_names] if df is not None else df
        return self.apply_multitreatment(Xdf, effect_interval_fun, *args, **kwargs)

    def effect_inference(self, df: pd.DataFrame, *args, **kwargs):
        """
        Inference (uncertainty) results produced by the underlying EconML estimator
        :param df: Features of the units to evaluate
        :param args: passed through to the underlying estimator
        :param kwargs: passed through to the underlying estimator
        """

        def effect_inference_fun(filtered_df, T0, T1, *args, **kwargs):
            return self.estimator.effect_inference(filtered_df, T0=T0, T1=T1, *args, **kwargs)

        Xdf = df[self._effect_modifier_names] if df is not None else df
        return self.apply_multitreatment(Xdf, effect_inference_fun, *args, **kwargs)

    def effect_tt(self, df: pd.DataFrame, treatment_value, *args, **kwargs):
        """
        Effect of the actual treatment that was applied to each unit
        ("effect of Treatment on the Treated")
        :param df: Features of the units to evaluate
        :param args: passed through to estimator.effect()
        :param kwargs: passed through to estimator.effect()
        """

        eff = self.effect(df[self._effect_modifier_names], *args, **kwargs).reshape((len(df), len(treatment_value)))

        out = np.zeros(len(df))
        treatment_value = parse_state(treatment_value)

        eff = np.reshape(eff, (len(df), len(treatment_value)))

        # For each unit, return the estimated effect of the treatment value
        # that was actually applied to the unit
        for c, col in enumerate(treatment_value):
            out[df[self._target_estimand.treatment_variable[0]] == col] = eff[
                df[self._target_estimand.treatment_variable[0]] == col, c
            ]
        return pd.Series(data=out, index=df.index)
