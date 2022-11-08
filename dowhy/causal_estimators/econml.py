import inspect
from importlib import import_module
from typing import Union, Any, List, Optional

import econml
import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.utils.api import parse_state
from dowhy.causal_identifier import IdentifiedEstimand


class Econml(CausalEstimator):
    """Wrapper class for estimators from the EconML library.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below. For init and fit
    parameters of each estimator, refer to the EconML docs.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        econml_methodname: str,
        test_significance: bool = False,
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
        :param econml_methodname: Fully qualified name of econml estimator
            class. For example, 'econml.dml.DML'
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
        # Required to ensure that self.method_params contains all the
        # parameters to create an object of this class
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
            econml_methodname=econml_methodname,
            **kwargs,
        )
        self._econml_methodname = econml_methodname
        self.logger.info("INFO: Using EconML Estimator")
        self.identifier_method = self._target_estimand.identifier_method

        self.estimator = None

    def fit(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        effect_modifier_names: Optional[List[str]] = None,
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
        self.set_data(data, treatment_name, outcome_name)
        self.set_effect_modifiers(effect_modifier_names)
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables().copy()

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

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

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
        self, treatment_value: Any = 1, control_value: Any = 0, confidence_intervals=False, target_units=None, **_
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        n_samples = self._treatment.shape[0]
        X = None  # Effect modifiers
        W = None  # common causes/ confounders
        Z = None  # Instruments
        Y = self._outcome
        T = self._treatment
        if self._effect_modifiers is not None and len(self._effect_modifiers) > 0:
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
        n_target_units = n_samples
        if X is not None:
            if type(target_units) is pd.DataFrame:
                X_test = target_units
            elif callable(target_units):
                filtered_rows = self._data.where(target_units)
                boolean_criterion = np.array(filtered_rows.notnull().iloc[:, 0])
                X_test = X[boolean_criterion]
            n_target_units = X_test.shape[0]

        # Changing shape to a list for a singleton value
        if type(control_value) is not list:
            control_value = [control_value]
        if type(treatment_value) is not list:
            treatment_value = [treatment_value]
        T0_test = np.repeat([control_value], n_target_units, axis=0)
        T1_test = np.repeat([treatment_value], n_target_units, axis=0)
        est = self.estimator.effect(X_test, T0=T0_test, T1=T1_test)
        ate = np.mean(est)

        self.effect_intervals = None
        self._confidence_intervals = confidence_intervals
        if self._confidence_intervals:
            self.effect_intervals = self.estimator.effect_interval(
                X_test, T0=T0_test, T1=T1_test, alpha=1 - self.confidence_level
            )
        estimate = CausalEstimate(
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

    def effect(self, df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        return self.estimator.effect(df[self._effect_modifier_names].values, *args, **kwargs)

    def effect_inference(self, df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        return self.estimator.effect_inference(df[self._effect_modifier_names].values, *args, **kwargs)
