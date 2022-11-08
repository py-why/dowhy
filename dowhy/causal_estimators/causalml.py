import inspect
from importlib import import_module
from typing import Any, List, Optional

import causalml
import pandas as pd

from dowhy.causal_estimator import CausalEstimate, CausalEstimator


class Causalml(CausalEstimator):
    """Wrapper class for estimators from the causalml library.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below. For specific
    parameters of each estimator, refer to the CausalML docs.

    """

    def __init__(
        self,
        identified_estimand,
        causalml_methodname,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        num_null_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction=CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level=CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates="auto",
        num_quantiles_to_discretize_cont_cols=CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
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
        # Required to ensure that self.method_params contains all the information
        # to create an object of this class
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
            causalml_methodname=causalml_methodname,
            **kwargs,
        )
        self._causalml_methodname = causalml_methodname
        # Add the identification method used in the estimator
        self.identifier_method = self._target_estimand.identifier_method
        self.logger.debug("The identifier method used {}".format(self.identifier_method))

        # Get the class corresponding the the estimator to be used
        estimator_class = self._get_causalml_class_object(self._causalml_methodname)
        # Initialize the object
        self.estimator = estimator_class(**self.method_params["init_params"])
        self.logger.info("INFO: Using CausalML Estimator")

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

        # Check the backdoor variables being used
        self.logger.debug("Back-door variables used:" + ",".join(self._target_estimand.get_backdoor_variables()))

        # Add the observed confounders and one hot encode the categorical variables
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if self._observed_common_causes_names:
            # Get the data of the unobserved confounders
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            # One hot encode the data if they are categorical
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = []

        # Check the instrumental variables involved
        self.logger.debug("Instrumental variables used:" + ",".join(self._target_estimand.instrumental_variables))

        # Perform the same actions as the above
        self._instrumental_variable_names = self._target_estimand.instrumental_variables
        if self._instrumental_variable_names:
            self._instrumental_variables = self._data[self._instrumental_variable_names]
            self._instrumental_variables = pd.get_dummies(self._instrumental_variables, drop_first=True)
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

    def estimate_effect(self, treatment_value: Any = 1, control_value: Any = 0, target_units=None, **_):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        X_names = self._observed_common_causes_names + self._effect_modifier_names

        # Both the outcome and the treatment have to be 1D arrays according to the CausalML API
        y_name = self._outcome_name
        treatment_name = self._treatment_name[0]  # As we have only one treatment variable
        # We want to pass 'v0' rather than ['v0'] to prevent a shape mismatch

        func_args = {"X": self._data[X_names], "y": self._data[y_name], "treatment": self._data[treatment_name]}

        arg_names = inspect.getfullargspec(self.estimator.estimate_ate)[0]
        matched_args = {arg: func_args[arg] for arg in func_args.keys() if arg in arg_names}
        self.logger.debug(matched_args)
        value_tuple = self.estimator.estimate_ate(**matched_args)

        # For CATEs
        arg_names = inspect.getfullargspec(self.estimator.fit_predict)[0]
        matched_args = {arg: func_args[arg] for arg in func_args.keys() if arg in arg_names}
        cate_estimates = self.estimator.fit_predict(**matched_args)

        estimate = CausalEstimate(
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
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
