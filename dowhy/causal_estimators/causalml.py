import inspect
import logging
from importlib import import_module
from typing import Any, List, Optional, Protocol, Union
from warnings import warn

import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class _CausalmlEstimator(Protocol):
    def estimate_ate(self, *args, **kwargs): ...

    def fit_predict(self, *args, **kwargs): ...


logger = logging.getLogger(__name__)


class Causalml(CausalEstimator):
    """Wrapper class for estimators from the causalml library.

    Supports additional parameters as listed below. For specific
    parameters of each estimator, refer to the CausalML docs.

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        causalml_estimator: Union[_CausalmlEstimator, str],
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
        :param causalml_methodname: Fully qualified name of causalml estimator
            class.
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
            causalml_estimator=causalml_estimator,
            **kwargs,
        )
        # Add the identification method used in the estimator
        self.identifier_method = self._target_estimand.identifier_method
        self.logger.debug("The identifier method used {}".format(self.identifier_method))

        if isinstance(causalml_estimator, str):
            warn(
                "Using a string to specify the value for causalml_estimator is now deprecated, please provide an instance of a causalml estimator object",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                estimator_class = self._get_causalml_class_object(causalml_estimator)
                self.estimator = estimator_class(**kwargs["init_params"])
            except ImportError:
                logger.error("You must install https://github.com/uber/causalml to use this functionality")
                raise
        else:
            self.estimator = causalml_estimator
        self.logger.info("INFO: Using CausalML Estimator")

    def fit(
        self,
        data: pd.DataFrame,
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
        self.reset_encoders()  # Forget any existing encoders
        self._set_effect_modifiers(data, effect_modifier_names)

        # Check the backdoor variables being used
        self.logger.debug("Adjustment set variables used:" + ",".join(self._target_estimand.get_adjustment_set()))

        # Add the observed confounders and one hot encode the categorical variables
        self._observed_common_causes_names = self._target_estimand.get_adjustment_set()
        if self._observed_common_causes_names:
            # Get the data of the unobserved confounders
            self._observed_common_causes = data[self._observed_common_causes_names]
            # One hot encode the data if they are categorical
            self._observed_common_causes = self._encode(self._observed_common_causes, "observed_common_causes")
        else:
            self._observed_common_causes = []

        # Check the instrumental variables involved
        self.logger.debug("Instrumental variables used:" + ",".join(self._target_estimand.instrumental_variables))

        # Perform the same actions as the above
        self._instrumental_variable_names = self._target_estimand.instrumental_variables
        if self._instrumental_variable_names:
            self._instrumental_variables = data[self._instrumental_variable_names]
            self._instrumental_variables = self._encode(self._instrumental_variables, "instrumental_variables")
        else:
            self._instrumental_variables = []

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def _get_causalml_class_object(self, module_method_name, *args, **kwargs):

        try:
            (module_name, _, class_name) = module_method_name.rpartition(".")
            estimator_module = import_module(module_name)
            estimator_class = getattr(estimator_module, class_name)

        except (AttributeError, AssertionError, ImportError):
            raise ImportError(
                "Error loading {}.{}. Double-check the method name and ensure that all causalml dependencies are installed.".format(
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
        treatment_value: value of the treatment variable for which the effect is to be estimated.
        control_value: value of the treatment variable that denotes its absence (usually 0)
        target_units: The units for which the treatment effect should be estimated.
                     It can be a DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
                     It can also be a lambda function that can be used as an index for the data (pandas DataFrame) to select the required rows.
        """
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        X_names = self._observed_common_causes_names + self._effect_modifier_names

        # Both the outcome and the treatment have to be 1D arrays according to the CausalML API
        y_name = self._target_estimand.outcome_variable[0]
        treatment_name = self._target_estimand.treatment_variable[0]  # As we have only one treatment variable
        # We want to pass 'v0' rather than ['v0'] to prevent a shape mismatch

        func_args = {"X": data[X_names], "y": data[y_name], "treatment": data[treatment_name]}

        arg_names = inspect.getfullargspec(self.estimator.estimate_ate)[0]
        matched_args = {arg: func_args[arg] for arg in func_args.keys() if arg in arg_names}
        self.logger.debug(matched_args)
        value_tuple = self.estimator.estimate_ate(**matched_args)

        # For CATEs
        arg_names = inspect.getfullargspec(self.estimator.fit_predict)[0]
        matched_args = {arg: func_args[arg] for arg in func_args.keys() if arg in arg_names}
        cate_estimates = self.estimator.fit_predict(**matched_args)

        estimate = CausalEstimate(
            data=data,
            treatment_name=treatment_name,
            outcome_name=self._target_estimand.outcome_variable[0],
            estimate=value_tuple[0],
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            cate_estimates=cate_estimates,
            effect_intervals=(value_tuple[1], value_tuple[2]),
            _estimator_object=self.estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO we are conditioning on a postive treatment
        # TODO create an expression corresponding to each estimator used
        var_list = estimand.treatment_variable + estimand.get_adjustment_set()
        expr += "+".join(var_list)
        return expr
