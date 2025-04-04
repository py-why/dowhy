import copy
import logging
from collections import namedtuple
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.utils import resample

import dowhy.interpreters as interpreters
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.utils.api import parse_state
from dowhy.utils.encoding import Encoders

logger = logging.getLogger(__name__)


class CausalEstimator:
    """Base class for an estimator of causal effect.

    Subclasses implement different estimation methods. All estimation methods are in the package "dowhy.causal_estimators"

    """

    # The default number of simulations for statistical testing
    DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST = 1000
    # The default number of simulations to obtain confidence intervals
    # This should be at least 399 for a 5% error rate:
    # https://www.econstor.eu/bitstream/10419/67820/1/587473266.pdf
    DEFAULT_NUMBER_OF_SIMULATIONS_CI = 399
    # The portion of the total size that should be taken each time to find the confidence intervals
    # 1 is the recommended value
    # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    # https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214
    DEFAULT_SAMPLE_SIZE_FRACTION = 1
    # The default Confidence Level
    DEFAULT_CONFIDENCE_LEVEL = 0.95
    # Number of quantiles to discretize continuous columns, for applying groupby
    NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS = 5
    # Prefix to add to temporary categorical variables created after discretization
    TEMP_CAT_COLUMN_PREFIX = "__categorical__"

    DEFAULT_NOTIMPLEMENTEDERROR_MSG = "not yet implemented for {0}. If you would this to be implemented in the next version, please raise an issue at https://github.com/microsoft/dowhy/issues"

    BootstrapEstimates = namedtuple("BootstrapEstimates", ["estimates", "params"])

    DEFAULT_INTERPRET_METHOD = ["textual_effect_interpreter"]

    # std args to be removed from locals() before being passed to args_dict
    _STD_INIT_ARGS = ("self", "__class__", "args", "kwargs")

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        **_,
    ):
        """Initializes an estimator with data and names of relevant variables.

        This method is called from the constructors of its child classes.

        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
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
        :returns: an instance of the estimator class.
        """
        self._target_estimand = identified_estimand

        self._significance_test = test_significance
        self._effect_strength_eval = evaluate_effect_strength
        self._confidence_intervals = confidence_intervals

        # Setting the default interpret method
        self.interpret_method = CausalEstimator.DEFAULT_INTERPRET_METHOD

        self.logger = logging.getLogger(__name__)

        # Check if some parameters were set, otherwise set to default values
        self.num_null_simulations = num_null_simulations
        self.num_simulations = num_simulations
        self.sample_size_fraction = sample_size_fraction
        self.confidence_level = confidence_level
        self.num_quantiles_to_discretize_cont_cols = num_quantiles_to_discretize_cont_cols
        # Estimate conditional estimates by default
        self.need_conditional_estimates = need_conditional_estimates

        self._bootstrap_estimates = None
        self._bootstrap_null_estimates = None

        self._encoders = Encoders()

    def reset_encoders(self):
        """
        Removes any reference to data encoders, causing them to be re-created on next `fit()`.

        It's important that data is consistently encoded otherwise models will produce inconsistent output.
        In particular, categorical variables are one-hot encoded; the mapping of original data values
        must be identical between model training/fitting and inference time.

        Encoders are reset when `fit()` is called again, as the data is assumed to have changed.

        A separate encoder is used for each subset of variables (treatment, common causes and effect modifiers).
        """
        self._encoders.reset()

    def _encode(self, data: pd.DataFrame, encoder_name: str):
        """
        Encodes categorical columns in the given data, returning a new dataframe containing
        all original data and the encoded columns. Numerical data is unchanged, categorical
        types are one-hot encoded. `encoder_name` identifies a specific encoder to be used
        if available, or created if not. The encoder can be reused in subsequent calls.

        :param data: Data to encode.
        :param encoder_name: The name for the encoder to be used.
        :returns: The encoded data.
        """
        return self._encoders.encode(data, encoder_name)

    def _set_effect_modifiers(self, data: pd.DataFrame, effect_modifier_names: Optional[List[str]] = None):
        """Sets the effect modifiers for the estimator
        Modifies need_conditional_estimates accordingly to effect modifiers value
        :param effect_modifiers: Variables on which to compute separate
            effects, or return a heterogeneous effect function. Not all
            methods support this currently.
        """
        self._effect_modifiers = effect_modifier_names
        if effect_modifier_names is not None:
            self._effect_modifier_names = [cname for cname in effect_modifier_names if cname in data.columns]
            if len(self._effect_modifier_names) > 0:
                self._effect_modifiers = data[self._effect_modifier_names]
                self._effect_modifiers = self._encode(self._effect_modifiers, "effect_modifiers")
                self.logger.debug("Effect modifiers: " + ",".join(self._effect_modifier_names))
            else:
                self._effect_modifier_names = []
        else:
            self._effect_modifier_names = []

        self.need_conditional_estimates = (
            self.need_conditional_estimates
            if self.need_conditional_estimates != "auto"
            else (self._effect_modifier_names and len(self._effect_modifier_names) > 0)
        )

    def _set_identified_estimand(self, new_identified_estimand):
        """Method used internally to change the target estimand (required by some refuters)

        :param new_identified_estimand: The new target_estimand to use
        """
        self._target_estimand = new_identified_estimand

    def get_new_estimator_object(
        self,
        identified_estimand,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=None,
    ):
        """Create a new estimator of the same type as the one passed in the estimate argument.

        Creates a new object with the identified_estimand

        :param identified_estimand: IdentifiedEstimand
            An instance of the identified estimand class that provides the information with
            respect to which causal pathways are employed when the treatment effects the outcome

        :returns: A new instance of the same estimator class that had generated the given estimate.
        """
        new_estimator = copy.deepcopy(self)

        new_estimator._target_estimand = identified_estimand
        new_estimator._test_significance = test_significance
        new_estimator._effect_strength_eval = evaluate_effect_strength
        new_estimator._confidence_intervals = (
            self._confidence_intervals if confidence_intervals is None else confidence_intervals
        )

        return new_estimator

    def estimate_effect_naive(self, data: pd.DataFrame):
        """
        :param data: Pandas dataframe to estimate effect
        """
        # TODO Only works for binary treatment
        df_withtreatment = data.loc[data[self._target_estimand.treatment_variable] == 1]
        df_notreatment = data.loc[data[self._target_estimand.treatment_variable] == 0]
        est = np.mean(df_withtreatment[self._target_estimand.outcome_variable]) - np.mean(
            df_notreatment[self._target_estimand.outcome_variable]
        )
        return CausalEstimate(data, None, None, est, None, control_value=0, treatment_value=1)

    def _estimate_effect_fn(self, data_df):
        """Function used in conditional effect estimation. This function is to be overridden by each child estimator.

        The overridden function should take in a dataframe as input and return the estimate for that data.
        """
        raise NotImplementedError(
            ("Conditional treatment effects are " + CausalEstimator.DEFAULT_NOTIMPLEMENTEDERROR_MSG).format(
                self.__class__
            )
        )

    def _estimate_conditional_effects(
        self, data: pd.DataFrame, estimate_effect_fn, effect_modifier_names=None, num_quantiles=None
    ):
        """Estimate conditional treatment effects. Common method for all estimators that utilizes a specific estimate_effect_fn implemented by each child estimator.

        If a numeric effect modifier is provided, it is discretized into quantile bins. If you would like a custom discretization, you can do so yourself: create a new column containing the discretized effect modifier and then include that column's name in the effect_modifier_names argument.

        :param data: Pandas dataframe to calculate the conditional effects
        :param estimate_effect_fn: Function that has a single parameter (a data frame) and returns the treatment effect estimate on that data.
        :param effect_modifier_names: Names of effect modifier variables over which the conditional effects will be estimated. If not provided, defaults to the effect modifiers specified during creation of the CausalEstimator object.
        :param num_quantiles: The number of quantiles into which a numeric effect modifier variable is discretized. Does not affect any categorical effect modifiers.

        :returns: A (multi-index) dataframe that provides separate effects for each value of the (discretized) effect modifiers.
        """
        # Defaulting to class default values if parameters are not provided
        if effect_modifier_names is None:
            effect_modifier_names = self._effect_modifier_names
        if num_quantiles is None:
            num_quantiles = self.num_quantiles_to_discretize_cont_cols
        # Checking that there is at least one effect modifier
        if not effect_modifier_names:
            raise ValueError("At least one effect modifier should be specified to compute conditional effects.")
        # Making sure that effect_modifier_names is a list
        effect_modifier_names = parse_state(effect_modifier_names)
        if not all(em in self._effect_modifier_names for em in effect_modifier_names):
            self.logger.warning(
                "At least one of the provided effect modifiers was not included while fitting the estimator. You may get incorrect results. To resolve, fit the estimator again by providing the updated effect modifiers in estimate_effect()."
            )
        # Making a copy since we are going to be changing effect modifier names
        effect_modifier_names = effect_modifier_names.copy()
        prefix = CausalEstimator.TEMP_CAT_COLUMN_PREFIX
        # For every numeric effect modifier, adding a temp categorical column
        for i in range(len(effect_modifier_names)):
            em = effect_modifier_names[i]
            if pd.api.types.is_numeric_dtype(data[em].dtypes):
                data[prefix + str(em)] = pd.qcut(data[em], num_quantiles, duplicates="drop")
                effect_modifier_names[i] = prefix + str(em)
        # Grouping by effect modifiers and computing effect separately
        by_effect_mods = data.groupby(effect_modifier_names)

        def cond_est_fn(x):
            return self._do(self._treatment_value, x) - self._do(self._control_value, x)

        conditional_estimates = by_effect_mods.apply(estimate_effect_fn)
        # Deleting the temporary categorical columns
        for em in effect_modifier_names:
            if em.startswith(prefix):
                data.pop(em)
        return conditional_estimates

    def _do(self, x, data_df=None):
        raise NotImplementedError(
            ("Do-operator is " + CausalEstimator.DEFAULT_NOTIMPLEMENTEDERROR_MSG).format(self.__class__)
        )

    def do(self, x, data_df=None):
        """Method that implements the do-operator.

        Given a value x for the treatment, returns the expected value of the outcome when the treatment is intervened to a value x.

        :param x: Value of the treatment
        :param data_df: Data on which the do-operator is to be applied.

        :returns: Value of the outcome when treatment is intervened/set to x.

        """
        est = self._do(x, data_df)
        return est

    def construct_symbolic_estimator(self, estimand):
        raise NotImplementedError(("Symbolic estimator string is ").format(self.__class__))

    def _generate_bootstrap_estimates(self, data: pd.DataFrame, num_bootstrap_simulations, sample_size_fraction):
        """Helper function to generate causal estimates over bootstrapped samples.

        :param num_bootstrap_simulations: Number of simulations for the bootstrap method.
        :param sample_size_fraction: Fraction of the dataset to be resampled.
        :returns: A collections.namedtuple containing a list of bootstrapped estimates and a dictionary containing parameters used for the bootstrap.
        """
        # The array that stores the results of all estimations
        simulation_results = np.zeros(num_bootstrap_simulations)

        # Find the sample size the proportion with the population size
        sample_size = int(sample_size_fraction * len(data))

        if sample_size > len(data):
            self.logger.warning("WARN: The sample size is greater than the data being sampled")

        self.logger.info("INFO: The sample size: {}".format(sample_size))
        self.logger.info("INFO: The number of simulations: {}".format(num_bootstrap_simulations))

        # Perform the set number of simulations
        for index in range(num_bootstrap_simulations):
            new_data = resample(data, n_samples=sample_size)
            new_estimator = self.get_new_estimator_object(
                self._target_estimand,
                # names of treatment and outcome
                test_significance=False,
                evaluate_effect_strength=False,
                confidence_intervals=False,
            )
            new_estimator.fit(
                new_data,
                effect_modifier_names=self._effect_modifier_names,
            )
            new_effect = new_estimator.estimate_effect(
                new_data,
                treatment_value=self._treatment_value,
                control_value=self._control_value,
                target_units=self._target_units,
            )
            simulation_results[index] = new_effect.value

        estimates = CausalEstimator.BootstrapEstimates(
            simulation_results,
            {"num_simulations": num_bootstrap_simulations, "sample_size_fraction": sample_size_fraction},
        )
        return estimates

    def _estimate_confidence_intervals_with_bootstrap(
        self,
        data: pd.DataFrame,
        estimate_value,
        confidence_level=None,
        num_simulations=None,
        sample_size_fraction=None,
    ):
        """
        Method to compute confidence interval using bootstrapped sampling.

        :param estimate_value: obtained estimate's value
        :param confidence_level: The level for which to compute CI (e.g., 95% confidence level translates to confidence_level=0.95)
        :param num_simulations: The number of simulations to be performed to get the bootstrap confidence intervals.
        :param sample_size_fraction: The fraction of the dataset to be resampled.
        :returns: confidence interval at the specified level.

        For more details on bootstrap or resampling statistics, refer to the following links:
        https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214
        """
        # Using class default parameters if not specified
        if num_simulations is None:
            num_simulations = self.num_simulations
        if sample_size_fraction is None:
            sample_size_fraction = self.sample_size_fraction

        # Checking if bootstrap_estimates are already computed
        if self._bootstrap_estimates is None:
            self._bootstrap_estimates = self._generate_bootstrap_estimates(data, num_simulations, sample_size_fraction)
        elif CausalEstimator.is_bootstrap_parameter_changed(self._bootstrap_estimates.params, locals()):
            # Checked if any parameter is changed from the previous std error estimate
            self._bootstrap_estimates = self._generate_bootstrap_estimates(data, num_simulations, sample_size_fraction)
        # Now use the data obtained from the simulations to get the value of the confidence estimates
        bootstrap_estimates = self._bootstrap_estimates.estimates
        # Get the variations of each bootstrap estimate and sort
        bootstrap_variations = [bootstrap_estimate - estimate_value for bootstrap_estimate in bootstrap_estimates]
        sorted_bootstrap_variations = np.sort(bootstrap_variations)

        # Now we take the (1-p)/2 th and the 1-(1-p)/2 th variations, where p is the chosen confidence level
        left_fraction = (1 - confidence_level) / 2
        right_fraction = 1 - left_fraction
        upper_bound_index = int(left_fraction * len(sorted_bootstrap_variations))
        lower_bound_index = int(right_fraction * len(sorted_bootstrap_variations))

        # Get the lower and upper bounds by subtracting the variations from the estimate
        lower_bound = estimate_value - sorted_bootstrap_variations[lower_bound_index]
        upper_bound = estimate_value - sorted_bootstrap_variations[upper_bound_index]
        return lower_bound, upper_bound

    def _estimate_confidence_intervals(self, confidence_level=None, method=None, **kwargs):
        """
        This method is to be overriden by the child classes, so that they
        can run a confidence interval estimation method suited to the specific
        causal estimator.
        """
        raise NotImplementedError(
            (
                "This method for estimating confidence intervals is "
                + CausalEstimator.DEFAULT_NOTIMPLEMENTEDERROR_MSG
                + " Meanwhile, you can try the bootstrap method (method='bootstrap') to estimate confidence intervals."
            ).format(self.__class__)
        )

    def estimate_confidence_intervals(
        self, data: pd.DataFrame, estimate_value, confidence_level=None, method=None, **kwargs
    ):
        """Find the confidence intervals corresponding to any estimator
        By default, this is done with the help of bootstrapped confidence intervals
        but can be overridden if the specific estimator implements other methods of estimating confidence intervals.

        If the method provided is not bootstrap, this function calls the implementation of the specific estimator.

        :param estimate_value: obtained estimate's value
        :param method: Method for estimating confidence intervals.
        :param confidence_level: The confidence level of the confidence intervals of the estimate.
        :param kwargs: Other optional args to be passed to the CI method.
        :returns: The obtained confidence interval.
        """
        if method is None:
            if self._confidence_intervals:
                method = self._confidence_intervals  # this is either True or methodname
            else:
                method = "default"
        confidence_intervals = None
        if confidence_level is None:
            confidence_level = self.confidence_level
        if method == "default" or method is True:  # user has not provided any method
            try:
                confidence_intervals = self._estimate_confidence_intervals(confidence_level, method=method, **kwargs)
            except NotImplementedError:
                confidence_intervals = self._estimate_confidence_intervals_with_bootstrap(
                    data, estimate_value, confidence_level, **kwargs
                )
        else:
            if method == "bootstrap":
                confidence_intervals = self._estimate_confidence_intervals_with_bootstrap(
                    data, estimate_value, confidence_level, **kwargs
                )
            else:
                confidence_intervals = self._estimate_confidence_intervals(confidence_level, method=method, **kwargs)
        return confidence_intervals

    def _estimate_std_error_with_bootstrap(self, data: pd.DataFrame, num_simulations=None, sample_size_fraction=None):
        """Compute standard error using the bootstrap method. Standard error
        and confidence intervals use the same parameter num_simulations for
        the number of bootstrap simulations.

        :param num_simulations: Number of bootstrapped samples.
        :param sample_size_fraction: Fraction of data to be resampled.
        :returns: Standard error of the obtained estimate.
        """
        # Use existing params, if new user defined params are not present
        if num_simulations is None:
            num_simulations = self.num_simulations
        if sample_size_fraction is None:
            sample_size_fraction = self.sample_size_fraction
        # Checking if bootstrap_estimates are already computed
        if self._bootstrap_estimates is None:
            self._bootstrap_estimates = self._generate_bootstrap_estimates(data, num_simulations, sample_size_fraction)
        elif CausalEstimator.is_bootstrap_parameter_changed(self._bootstrap_estimates.params, locals()):
            # Check if any parameter is changed from the previous std error estimate
            self._bootstrap_estimates = self._generate_bootstrap_estimates(data, num_simulations, sample_size_fraction)

        std_error = np.std(self._bootstrap_estimates.estimates)
        return std_error

    def _estimate_std_error(self, method=None, **kwargs):
        """
        This method is to be overriden by the child classes, so that they
        can run a standard error estimation method suited to the specific
        causal estimator.
        """
        raise NotImplementedError(
            (
                "This method for estimating standard errors is "
                + CausalEstimator.DEFAULT_NOTIMPLEMENTEDERROR_MSG
                + " Meanwhile, you can try the bootstrap method (method='bootstrap') to estimate standard errors."
            ).format(self.__class__)
        )

    def estimate_std_error(self, data: pd.DataFrame, method=None, **kwargs):
        """Compute standard error of an obtained causal estimate.

        :param method: Method for computing the standard error.
        :param kwargs: Other optional parameters to be passed to the estimating method.
        :returns: Standard error of the causal estimate.
        """
        if method is None:
            if self._confidence_intervals:
                method = self._confidence_intervals
            else:
                method = "default"
        std_error = None
        if method == "default" or method is True:  # user has not provided any method
            try:
                std_error = self._estimate_std_error(method, **kwargs)
            except NotImplementedError:
                std_error = self._estimate_std_error_with_bootstrap(data, **kwargs)
        else:
            if method == "bootstrap":
                std_error = self._estimate_std_error_with_bootstrap(data, **kwargs)
            else:
                std_error = self._estimate_std_error(method, **kwargs)
        return std_error

    def _test_significance_with_bootstrap(self, data: pd.DataFrame, estimate_value, num_null_simulations=None):
        """Test statistical significance of an estimate using the bootstrap method.

        :param estimate_value: Obtained estimate's value
        :param num_null_simulations: Number of simulations for the null hypothesis
        :returns: p-value of the statistical significance test.
        """
        # Use existing params, if new user defined params are not present
        if num_null_simulations is None:
            num_null_simulations = self.num_null_simulations
        do_retest = self._bootstrap_null_estimates is None or CausalEstimator.is_bootstrap_parameter_changed(
            self._bootstrap_null_estimates.params, locals()
        )
        if do_retest:
            null_estimates = np.zeros(num_null_simulations)
            new_estimand = copy.deepcopy(self._target_estimand)
            new_estimand.outcome_variable = ["dummy_outcome"]
            for i in range(num_null_simulations):
                new_outcome = np.random.permutation(data[self._target_estimand.outcome_variable])
                new_data = data.assign(dummy_outcome=new_outcome)
                new_estimator = self.get_new_estimator_object(
                    new_estimand,
                    test_significance=False,
                    evaluate_effect_strength=False,
                    confidence_intervals=False,
                )
                new_estimator.fit(
                    data=new_data,
                    effect_modifier_names=self._effect_modifier_names,
                )

                new_effect = new_estimator.estimate_effect(
                    new_data,
                    target_units=self._target_units,
                )
                null_estimates[i] = new_effect.value
            self._bootstrap_null_estimates = CausalEstimator.BootstrapEstimates(
                null_estimates, {"num_null_simulations": num_null_simulations, "sample_size_fraction": 1}
            )

        # Processing the null hypothesis estimates
        sorted_null_estimates = np.sort(self._bootstrap_null_estimates.estimates)
        self.logger.debug("Null estimates: {0}".format(sorted_null_estimates))
        median_estimate = sorted_null_estimates[int(num_null_simulations / 2)]
        # Doing a two-sided test
        if estimate_value > median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate_value, side="left")
            p_value = 1 - (estimate_index / num_null_simulations)
        if estimate_value <= median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate_value, side="right")
            p_value = estimate_index / num_null_simulations
        # If the estimate_index is 0, it depends on the number of simulations
        if p_value == 0:
            p_value = (0, 1 / len(sorted_null_estimates))  # a tuple determining the range.
        elif p_value == 1:
            p_value = (1 - 1 / len(sorted_null_estimates), 1)
        signif_dict = {"p_value": p_value}
        return signif_dict

    def _test_significance(self, estimate_value, method=None, **kwargs):
        """
        This method is to be overriden by the child classes, so that they
        can run a significance test suited to the specific
        causal estimator.
        """
        raise NotImplementedError(
            (
                "This method for testing statistical significance is "
                + CausalEstimator.DEFAULT_NOTIMPLEMENTEDERROR_MSG
                + " Meanwhile, you can try the bootstrap method (method='bootstrap') to test statistical significance."
            ).format(self.__class__)
        )

    def test_significance(self, data: pd.DataFrame, estimate_value, method=None, **kwargs):
        """Test statistical significance of obtained estimate.

        By default, uses resampling to create a non-parametric significance test.
        A general procedure. Individual child estimators can implement different methods.
        If the method name is different from "bootstrap", this function calls the
        implementation of the child estimator.

        :param self: object instance of class Estimator
        :param estimate_value: obtained estimate's value
        :param method: Method for checking statistical significance

        :returns: p-value from the significance test

        """
        if method is None:
            if self._significance_test:
                method = self._significance_test  # this is either True or methodname
            else:
                method = "default"
        signif_dict = None
        if method == "default" or method is True:  # user has not provided any method
            try:
                signif_dict = self._test_significance(estimate_value, method, **kwargs)
            except NotImplementedError:
                signif_dict = self._test_significance_with_bootstrap(data, estimate_value, **kwargs)
        else:
            if method == "bootstrap":
                signif_dict = self._test_significance_with_bootstrap(data, estimate_value, **kwargs)
            else:
                signif_dict = self._test_significance(estimate_value, method, **kwargs)
        return signif_dict

    def evaluate_effect_strength(self, data: pd.DataFrame, estimate):
        fraction_effect_explained = self._evaluate_effect_strength(data, estimate, method="fraction-effect")
        # Need to test r-squared before supporting
        # effect_r_squared = self._evaluate_effect_strength(estimate, method="r-squared")
        strength_dict = {
            "fraction-effect": fraction_effect_explained
            #       'r-squared': effect_r_squared
        }
        return strength_dict

    def _evaluate_effect_strength(self, data: pd.DataFrame, estimate, method="fraction-effect"):
        supported_methods = ["fraction-effect"]
        if method not in supported_methods:
            raise NotImplementedError("This method is not supported for evaluating effect strength")
        if method == "fraction-effect":
            naive_obs_estimate = self.estimate_effect_naive(data)
            self.logger.debug(estimate.value, naive_obs_estimate.value)
            fraction_effect_explained = estimate.value / naive_obs_estimate.value
            return fraction_effect_explained
        # elif method == "r-squared":
        #    outcome_mean = np.mean(self._outcome)
        #    total_variance = np.sum(np.square(self._outcome - outcome_mean))
        # Assuming a linear model with one variable: the treatment
        # Currently only works for continuous y
        #    causal_model = outcome_mean + estimate.value*self._treatment
        #    squared_residual = np.sum(np.square(self._outcome - causal_model))
        #    r_squared = 1 - (squared_residual/total_variance)
        #    return r_squared
        else:
            return None

    def update_input(self, treatment_value, control_value, target_units):
        self._control_value = control_value
        self._treatment_value = treatment_value
        self._target_units = target_units

    @staticmethod
    def is_bootstrap_parameter_changed(bootstrap_estimates_params, given_params):
        """Check whether parameters of the bootstrap have changed.

        This is an efficiency method that checks if fresh resampling of the bootstrap samples is required.
        Returns True if parameters have changed and resampling should be done again.

        :param bootstrap_estimates_params: A dictionary of parameters for the current bootstrap samples
        :param given_params: A dictionary of parameters passed by the user
        :returns: A binary flag denoting whether the parameters are different.
        """
        is_any_parameter_changed = False
        for prm, val in bootstrap_estimates_params.items():
            given_val = given_params.get(prm, None)
            if given_val is not None and given_val != val:
                is_any_parameter_changed = True
                break
        return is_any_parameter_changed

    def target_units_tostr(self):
        s = ""
        if type(self._target_units) is str:
            s += self._target_units
        elif callable(self._target_units):
            s += "Data subset defined by a function"
        elif isinstance(self._target_units, pd.DataFrame):
            s += "Data subset provided as a data frame"
        return s

    def signif_results_tostr(self, signif_results):
        s = ""
        pval = signif_results["p_value"]
        if type(pval) is tuple:
            s += "[{0}, {1}]".format(pval[0], pval[1])
        else:
            s += "{0}".format(pval)
        return s


def estimate_effect(
    data: pd.DataFrame,
    treatment: Union[str, List[str]],
    outcome: Union[str, List[str]],
    identifier_name: str,
    estimator: CausalEstimator,
    control_value: int = 0,
    treatment_value: int = 1,
    target_units: str = "ate",
    effect_modifiers: List[str] = None,
    fit_estimator: bool = True,
    method_params: Optional[Dict] = None,
):
    """Estimate the identified causal effect.

    In addition, you can directly call any of the EconML estimation methods. The convention is "backdoor.econml.path-to-estimator-class". For example, for the double machine learning estimator ("DML" class) that is located inside "dml" module of EconML, you can use the method name, "backdoor.econml.dml.DML". CausalML estimators can also be called. See `this demo notebook <https://py-why.github.io/dowhy/example_notebooks/dowhy-conditional-treatment-effects.html>`_.

    :param treatment: Name of the treatment
    :param outcome: Name of the outcome
    :param identified_estimand: a probability expression
        that represents the effect to be estimated. Output of
        CausalModel.identify_effect method
    :param estimator: Instance of a CausalEstimator to use
    :param control_value: Value of the treatment in the control group, for effect estimation.  If treatment is multi-variate, this can be a list.
    :param treatment_value: Value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list.
    :param target_units: (Experimental) The units for which the treatment effect should be estimated. This can be of three types. (1) a string for common specifications of target units (namely, "ate", "att" and "atc"), (2) a lambda function that can be used as an index for the data (pandas DataFrame), or (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
    :param effect_modifiers: Names of effect modifier variables can be (optionally) specified here too, since they do not affect identification. If None, the effect_modifiers from the CausalModel are used.
    :param fit_estimator: Boolean flag on whether to fit the estimator.
        Setting it to False is useful to estimate the effect on new data using a previously fitted estimator.
    :returns: An instance of the CausalEstimate class, containing the causal effect estimate
        and other method-dependent information
    """

    if effect_modifiers is None:
        effect_modifiers = []

    treatment = parse_state(treatment)
    outcome = parse_state(outcome)
    causal_estimator_class = estimator.__class__

    identified_estimand = estimator._target_estimand
    identified_estimand.set_identifier_method(identifier_name)

    if identified_estimand.no_directed_path:
        logger.warning("No directed path from {0} to {1}.".format(treatment, outcome))
        return CausalEstimate(
            None, None, None, 0, identified_estimand, None, control_value=control_value, treatment_value=treatment_value
        )
    # Check if estimator's target estimand is identified
    elif identified_estimand.estimands[identifier_name] is None:
        logger.error("No valid identified estimand available.")
        return CausalEstimate(
            None, None, None, None, None, None, control_value=control_value, treatment_value=treatment_value
        )

    if fit_estimator:
        estimator.fit(
            data=data,
            effect_modifier_names=effect_modifiers,
            **method_params["fit_params"] if "fit_params" in method_params else {},
        )

    estimate = estimator.estimate_effect(
        data,
        treatment_value=treatment_value,
        control_value=control_value,
        target_units=target_units,
        confidence_intervals=estimator._confidence_intervals,
    )

    if estimator._significance_test:
        estimator.test_significance(data, estimate.value, method=estimator._significance_test)
    if estimator._confidence_intervals:
        estimator.estimate_confidence_intervals(
            data,
            estimate.value,
            confidence_level=estimator.confidence_level,
            method=estimator._confidence_intervals,
        )
    if estimator._effect_strength_eval:
        effect_strength_dict = estimator.evaluate_effect_strength(data, estimate)
        estimate.add_effect_strength(effect_strength_dict)

    # Store parameters inside estimate object for refutation methods
    # TODO: This add_params needs to move to the estimator class
    # inside estimate_effect and estimate_conditional_effect
    estimate.add_params(
        estimand_type=identified_estimand.estimand_type,
        estimator_class=causal_estimator_class,
        test_significance=estimator._significance_test,
        evaluate_effect_strength=estimator._effect_strength_eval,
        confidence_intervals=estimator._confidence_intervals,
        target_units=target_units,
        effect_modifiers=effect_modifiers,
    )

    return estimate


class CausalEstimate:
    """Class for the estimate object that every causal estimator returns"""

    def __init__(
        self,
        data,
        treatment_name,
        outcome_name,
        estimate,
        target_estimand,
        realized_estimand_expr,
        control_value,
        treatment_value,
        conditional_estimates=None,
        **kwargs,
    ):
        # TODO: Remove _data, _treatment_name and _outcome_name from this object
        # we save them here to enable the methods that required these properties saved in the estimator
        # eventually we should call those methods and just save the results in this object
        # instead of having this object invoke the estimator methods with the data.
        self._data = data
        self._treatment_name = treatment_name
        self._outcome_name = outcome_name
        self.value = estimate
        self.target_estimand = target_estimand
        self.realized_estimand_expr = realized_estimand_expr
        self.control_value = control_value
        self.treatment_value = treatment_value
        self.conditional_estimates = conditional_estimates
        self.params = kwargs
        if self.params is not None:
            for key, value in self.params.items():
                setattr(self, key, value)

        self.effect_strength = None

    def add_estimator(self, estimator_instance):
        self.estimator = estimator_instance

    def add_effect_strength(self, strength_dict):
        self.effect_strength = strength_dict

    def add_params(self, **kwargs):
        self.params.update(kwargs)

    def get_confidence_intervals(self, confidence_level=None, method=None, **kwargs):
        """Get confidence intervals of the obtained estimate.

        By default, this is done with the help of bootstrapped confidence intervals
        but can be overridden if the specific estimator implements other methods of estimating confidence intervals.

        If the method provided is not bootstrap, this function calls the implementation of the specific estimator.

        :param method: Method for estimating confidence intervals.
        :param confidence_level: The confidence level of the confidence intervals of the estimate.
        :param kwargs: Other optional args to be passed to the CI method.
        :returns: The obtained confidence interval.
        """
        confidence_intervals = self.estimator.estimate_confidence_intervals(
            data=self._data,
            estimate_value=self.value,
            confidence_level=confidence_level,
            method=method,
            **kwargs,
        )
        return confidence_intervals

    def get_standard_error(self, method=None, **kwargs):
        """Get standard error of the obtained estimate.

        By default, this is done with the help of bootstrapped standard errors
        but can be overridden if the specific estimator implements other methods of estimating standard error.

        If the method provided is not bootstrap, this function calls the implementation of the specific estimator.

        :param method: Method for computing the standard error.
        :param kwargs: Other optional parameters to be passed to the estimating method.
        :returns: Standard error of the causal estimate.

        """
        std_error = self.estimator.estimate_std_error(self._data, method=method, **kwargs)
        return std_error

    def test_stat_significance(self, method=None, **kwargs):
        """Test statistical significance of the estimate obtained.

        By default, uses resampling to create a non-parametric significance test.
        Individual child estimators can implement different methods.
        If the method name is different from "bootstrap", this function calls the
        implementation of the child estimator.

        :param method: Method for checking statistical significance
        :param kwargs: Other optional parameters to be passed to the estimating method.

        :returns: p-value from the significance test
        """
        signif_results = self.estimator.test_significance(self._data, self.value, method=method, **kwargs)
        return {"p_value": signif_results["p_value"]}

    def estimate_conditional_effects(
        self, effect_modifiers=None, num_quantiles=CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS
    ):
        """Estimate treatment effect conditioned on given variables.

        If a numeric effect modifier is provided, it is discretized into quantile bins. If you would like a custom discretization, you can do so yourself: create a new column containing the discretized effect modifier and then include that column's name in the effect_modifier_names argument.

        :param effect_modifiers: Names of effect modifier variables over which the conditional effects will be estimated. If not provided, defaults to the effect modifiers specified during creation of the CausalEstimator object.
        :param num_quantiles: The number of quantiles into which a numeric effect modifier variable is discretized. Does not affect any categorical effect modifiers.

        :returns: A (multi-index) dataframe that provides separate effects for each value of the (discretized) effect modifiers.
        """
        return self.estimator._estimate_conditional_effects(
            self.estimator._estimate_effect_fn, effect_modifiers, num_quantiles
        )

    def interpret(self, method_name=None, **kwargs):
        """Interpret the causal estimate.

        :param method_name: Method used (string) or a list of methods. If None, then the default for the specific estimator is used.
        :param kwargs:: Optional parameters that are directly passed to the interpreter method.

        :returns: None

        """
        if method_name is None:
            method_name = self.estimator.interpret_method
        method_name_arr = parse_state(method_name)

        for method in method_name_arr:
            interpreter = interpreters.get_class_object(method)
            interpreter(self, **kwargs).interpret(self._data)

    def __str__(self):
        s = "*** Causal Estimate ***\n"
        # No estimand was identified (identification failed)
        if self.target_estimand is None:
            return "Estimation failed! No relevant identified estimand available for this estimation method."
        s += "\n## Identified estimand\n{0}".format(self.target_estimand.__str__(only_target_estimand=True))
        s += "\n## Realized estimand\n{0}".format(self.realized_estimand_expr)
        if hasattr(self, "estimator"):
            s += "\nTarget units: {0}\n".format(self.estimator.target_units_tostr())
        s += "\n## Estimate\n"
        s += "Mean value: {0}\n".format(self.value)
        s += ""
        if hasattr(self, "cate_estimates"):
            s += "Effect estimates: {0}\n".format(self.cate_estimates)
        if hasattr(self, "estimator"):
            if self.estimator._significance_test:
                s += "p-value: {0}\n".format(self.estimator.signif_results_tostr(self.test_stat_significance()))
            if self.estimator._confidence_intervals:
                s += "{0}% confidence interval: {1}\n".format(
                    100 * self.estimator.confidence_level, self.get_confidence_intervals()
                )
        if self.conditional_estimates is not None:
            s += "### Conditional Estimates\n"
            s += str(self.conditional_estimates)
        if self.effect_strength is not None:
            s += "\n## Effect Strength\n"
            s += "Change in outcome attributable to treatment: {}\n".format(self.effect_strength["fraction-effect"])
            # s += "Variance in outcome explained by treatment: {}\n".format(self.effect_strength["r-squared"])
        return s


class RealizedEstimand(object):
    def __init__(self, identified_estimand, estimator_name):
        self.treatment_variable = identified_estimand.treatment_variable
        self.outcome_variable = identified_estimand.outcome_variable
        self.backdoor_variables = identified_estimand.get_backdoor_variables()
        self.general_adjustment_variables = identified_estimand.get_general_adjustment_variables()
        self.instrumental_variables = identified_estimand.instrumental_variables
        self.estimand_type = identified_estimand.estimand_type
        self.estimand_expression = None
        self.assumptions = None
        self.estimator_name = estimator_name

    def update_assumptions(self, estimator_assumptions):
        self.assumptions = estimator_assumptions

    def update_estimand_expression(self, estimand_expression):
        self.estimand_expression = estimand_expression

    def __str__(self):
        s = "Realized estimand: {0}\n".format(self.estimator_name)
        s += "Realized estimand type: {0}\n".format(self.estimand_type)
        s += "Estimand expression:\n{0}\n".format(sp.pretty(self.estimand_expression))
        j = 1
        for ass_name, ass_str in self.assumptions.items():
            s += "Estimand assumption {0}, {1}: {2}\n".format(j, ass_name, ass_str)
            j += 1
        return s
