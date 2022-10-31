import copy
import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefutation, CausalRefuter, choose_variables
from dowhy.causal_refuters.evalue_sensitivity_analyzer import EValueSensitivityAnalyzer
from dowhy.causal_refuters.linear_sensitivity_analyzer import LinearSensitivityAnalyzer
from dowhy.causal_refuters.non_parametric_sensitivity_analyzer import NonParametricSensitivityAnalyzer
from dowhy.causal_refuters.partial_linear_sensitivity_analyzer import PartialLinearSensitivityAnalyzer

logger = logging.getLogger(__name__)


DEFAULT_CONVERGENCE_THRESHOLD = 0.1
DEFAULT_C_STAR_MAX = 1000


class AddUnobservedCommonCause(CausalRefuter):

    """Add an unobserved confounder for refutation.

    AddUnobservedCommonCause class supports three methods:
        1) Simulation of an unobserved confounder
        2) Linear partial R2 : Sensitivity Analysis for linear models.
        3) Non-Parametric partial R2 based : Sensitivity Analyis for non-parametric models.

    Supports additional parameters that can be specified in the refute_estimate() method.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters required for the refuter.

        For direct_simulation, if effect_strength_on_treatment or effect_strength_on_outcome is not
        given, it is calculated automatically as a range between the
        minimum and maximum effect strength of observed confounders on treatment
        and outcome respectively.

        :param simulation_method: The method to use for simulating effect of unobserved confounder. Possible values are ["direct-simulation", "linear-partial-R2", "non-parametric-partial-R2", "e-value"].
        :param confounders_effect_on_treatment: str : The type of effect on the treatment due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
        :param confounders_effect_on_outcome: str : The type of effect on the outcome due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
        :param effect_strength_on_treatment: float, numpy.ndarray: [Used when simulation_method="direct-simulation"] Strength of the confounder's effect on treatment. When confounders_effect_on_treatment is linear,  it is the regression coefficient. When the confounders_effect_on_treatment is binary flip, it is the probability with which effect of unobserved confounder can invert the value of the treatment.
        :param effect_strength_on_outcome: float, numpy.ndarray: Strength of the confounder's effect on outcome. Its interpretation depends on confounders_effect_on_outcome and the simulation_method. When simulation_method is direct-simulation, for a linear effect it behaves like the regression coefficient and for a binary flip, it is the probability with which it can invert the value of the outcome.
        :param partial_r2_confounder_treatment: float, numpy.ndarray: [Used when simulation_method is linear-partial-R2 or non-parametric-partial-R2] Partial R2 of the unobserved confounder wrt the treatment conditioned on the observed confounders. Only in the case of general non-parametric-partial-R2, it is the fraction of variance in the reisz representer that is explained by the unobserved confounder; specifically  (1-r), where r is the ratio of variance of reisz representer, alpha^2, based on observed confounders and that based on all confounders.
        :param partial_r2_confounder_outcome: float, numpy.ndarray: [Used when simulation_method is linear-partial-R2 or non-parametric-partial-R2] Partial R2 of the unobserved confounder wrt the outcome conditioned on the treatment and observed confounders.
        :param frac_strength_treatment: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on treatment. Defaults to 1.
        :param frac_strength_outcome: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on outcome. Defaults to 1.
        :param plotmethod: string: Type of plot to be shown. If None, no plot is generated. This parameter is used only only when more than one treatment confounder effect values or outcome confounder effect values are provided. Default is "colormesh". Supported values are "contour", "colormesh" when more than one value is provided for both confounder effect value parameters; "line" when provided for only one of them.
        :param percent_change_estimate: It is the percentage of reduction of treatment estimate that could alter the results (default = 1).
                                        if percent_change_estimate = 1, the robustness value describes the strength of association of confounders with treatment and outcome in order to reduce the estimate by 100% i.e bring it down to 0. (relevant only for Linear Sensitivity Analysis, ignore for rest)
        :param confounder_increases_estimate: True implies that confounder increases the absolute value of estimate and vice versa. (Default = False). (relevant only for Linear Sensitivity Analysis, ignore for rest)
        :param benchmark_common_causes: names of variables for bounding strength of confounders. (relevant only for partial-r2 based simulation methods)
        :param significance_level: confidence interval for statistical inference(default = 0.05). (relevant only for partial-r2 based simulation methods)
        :param null_hypothesis_effect: assumed effect under the null hypothesis. (relevant only for linear-partial-R2, ignore for rest)
        :param plot_estimate: Generate contour plot for estimate while performing sensitivity analysis. (default = True).
                              (relevant only for partial-r2 based simulation methods)
        :param num_splits: number of splits for cross validation. (default = 5). (relevant only for non-parametric-partial-R2 simulation method)
        :param shuffle_data : shuffle data or not before splitting into folds (default = False). (relevant only for non-parametric-partial-R2 simulation method)
        :param shuffle_random_seed: seed for randomly shuffling data. (relevant only for non-parametric-partial-R2 simulation method)
        :param alpha_s_estimator_param_list: list of dictionaries with parameters for finding alpha_s. (relevant only for non-parametric-partial-R2 simulation method)
        :param g_s_estimator_list: list of estimator objects for finding g_s. These objects should have fit() and predict() functions implemented. (relevant only for non-parametric-partial-R2 simulation method)
        :param g_s_estimator_param_list: list of dictionaries with parameters for tuning respective estimators in "g_s_estimator_list". The order of the dictionaries in the list should be consistent with the estimator objects order in "g_s_estimator_list". (relevant only for non-parametric-partial-R2 simulation method)
        """
        super().__init__(*args, **kwargs)
        self.simulation_method = kwargs["simulation_method"] if "simulation_method" in kwargs else "direct-simulation"
        self.effect_on_t = (
            kwargs["confounders_effect_on_treatment"] if "confounders_effect_on_treatment" in kwargs else "binary_flip"
        )
        self.effect_on_y = (
            kwargs["confounders_effect_on_outcome"] if "confounders_effect_on_outcome" in kwargs else "linear"
        )
        if self.simulation_method == "direct-simulation":
            self.kappa_t = kwargs["effect_strength_on_treatment"] if "effect_strength_on_treatment" in kwargs else None
            self.kappa_y = kwargs["effect_strength_on_outcome"] if "effect_strength_on_outcome" in kwargs else None
        elif self.simulation_method in ["linear-partial-R2", "non-parametric-partial-R2"]:
            self.kappa_t = (
                kwargs["partial_r2_confounder_treatment"] if "partial_r2_confounder_treatment" in kwargs else None
            )
            self.kappa_y = (
                kwargs["partial_r2_confounder_outcome"] if "partial_r2_confounder_outcome" in kwargs else None
            )
        elif self.simulation_method == "e-value":
            pass
        else:
            raise ValueError(
                "simulation method is not supported. Try direct-simulation, linear-partial-R2, non-parametric-partial-R2, or e-value"
            )
        self.frac_strength_treatment = (
            kwargs["effect_fraction_on_treatment"] if "effect_fraction_on_treatment" in kwargs else 1
        )
        self.frac_strength_outcome = (
            kwargs["effect_fraction_on_outcome"] if "effect_fraction_on_outcome" in kwargs else 1
        )

        self.plotmethod = kwargs["plotmethod"] if "plotmethod" in kwargs else "colormesh"
        self.percent_change_estimate = kwargs["percent_change_estimate"] if "percent_change_estimate" in kwargs else 1.0
        self.significance_level = kwargs["significance_level"] if "significance_level" in kwargs else 0.05
        self.confounder_increases_estimate = (
            kwargs["confounder_increases_estimate"] if "confounder_increases_estimate" in kwargs else False
        )
        self.benchmark_common_causes = (
            kwargs["benchmark_common_causes"] if "benchmark_common_causes" in kwargs else None
        )
        self.null_hypothesis_effect = kwargs["null_hypothesis_effect"] if "null_hypothesis_effect" in kwargs else 0
        self.plot_estimate = kwargs["plot_estimate"] if "plot_estimate" in kwargs else True
        self.num_splits = kwargs["num_splits"] if "num_splits" in kwargs else 5
        self.shuffle_data = kwargs["shuffle_data"] if "shuffle_data" in kwargs else False
        self.shuffle_random_seed = kwargs["shuffle_random_seed"] if "shuffle_random_seed" in kwargs else None
        self.alpha_s_estimator_param_list = (
            kwargs["alpha_s_estimator_param_list"] if "alpha_s_estimator_param_list" in kwargs else None
        )
        self.alpha_s_estimator_list = kwargs["alpha_s_estimator_list"] if "alpha_s_estimator_list" in kwargs else None
        self.g_s_estimator_list = kwargs["g_s_estimator_list"] if "g_s_estimator_list" in kwargs else None
        self.g_s_estimator_param_list = (
            kwargs["g_s_estimator_param_list"] if "g_s_estimator_param_list" in kwargs else None
        )
        self.plugin_reisz = kwargs["plugin_reisz"] if "plugin_reisz" in kwargs else False
        self.logger = logging.getLogger(__name__)

    def refute_estimate(self, show_progress_bar=False):
        if self.simulation_method == "linear-partial-R2":
            return sensitivity_linear_partial_r2(
                self._data,
                self._estimate,
                self._treatment_name,
                self.frac_strength_treatment,
                self.frac_strength_outcome,
                self.percent_change_estimate,
                self.benchmark_common_causes,
                self.significance_level,
                self.null_hypothesis_effect,
                self.plot_estimate,
            )
        elif self.simulation_method == "non-parametric-partial-R2":
            return sensitivity_non_parametric_partial_r2(
                self._estimate,
                self.kappa_t,
                self.kappa_y,
                self.frac_strength_treatment,
                self.frac_strength_outcome,
                self.benchmark_common_causes,
                self.plot_estimate,
                self.alpha_s_estimator_list,
                self.alpha_s_estimator_param_list,
                self.g_s_estimator_list,
                self.g_s_estimator_param_list,
                self.plugin_reisz,
            )
        elif self.simulation_method == "e-value":
            return sensitivity_e_value(
                self._data,
                self._target_estimand,
                self._estimate,
                self._treatment_name,
                self._outcome_name,
                self.plot_estimate,
            )
        elif self.simulation_method == "direct-simulation":
            refute = sensitivity_simulation(
                self._data,
                self._target_estimand,
                self._estimate,
                self._treatment_name,
                self._outcome_name,
                self.kappa_t,
                self.kappa_y,
                self.effect_on_t,
                self.effect_on_y,
                self.frac_strength_treatment,
                self.frac_strength_outcome,
                self.plotmethod,
                show_progress_bar,
            )
            refute.add_refuter(self)
            return refute

    def include_simulated_confounder(
        self, convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD, c_star_max=DEFAULT_C_STAR_MAX
    ):
        return include_simulated_confounder(
            self._data,
            self._treatment_name,
            self._outcome_name,
            self.kappa_t,
            self.kappa_y,
            self._variables_of_interest,
            convergence_threshold,
            c_star_max,
        )


def _infer_default_kappa_t(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    treatment_name: List[str],
    effect_on_t: str,
    frac_strength_treatment: float,
    len_kappa_t: int = 10,
):
    """Infer default effect strength of simulated confounder on treatment."""
    observed_common_causes_names = target_estimand.get_backdoor_variables()
    if len(observed_common_causes_names) > 0:
        observed_common_causes = data[observed_common_causes_names]
        observed_common_causes = pd.get_dummies(observed_common_causes, drop_first=True)
    else:
        raise ValueError(
            "There needs to be at least one common cause to"
            + "automatically compute the default value of kappa_t."
            + " Provide a value for kappa_t"
        )
    t = data[treatment_name]
    # Standardizing the data
    observed_common_causes = StandardScaler().fit_transform(observed_common_causes)
    if effect_on_t == "binary_flip":
        # Fit a model containing all confounders and compare predictions
        # using all features compared to all features except a given
        # confounder.
        tmodel = LogisticRegression().fit(observed_common_causes, t)
        tpred = tmodel.predict(observed_common_causes).astype(int)
        flips = []
        for i in range(observed_common_causes.shape[1]):
            oldval = np.copy(observed_common_causes[:, i])
            observed_common_causes[:, i] = 0
            tcap = tmodel.predict(observed_common_causes).astype(int)
            observed_common_causes[:, i] = oldval
            flips.append(np.sum(abs(tcap - tpred)) / tpred.shape[0])
        min_coeff, max_coeff = min(flips), max(flips)
    elif effect_on_t == "linear":
        # Estimating the regression coefficient from standardized features to t
        corrcoef_var_t = np.corrcoef(observed_common_causes, t, rowvar=False)[-1, :-1]
        std_dev_t = np.std(t)[0]
        max_coeff = max(corrcoef_var_t) * std_dev_t
        min_coeff = min(corrcoef_var_t) * std_dev_t
    else:
        raise NotImplementedError("'" + effect_on_t + "' method not supported for confounders' effect on treatment")

    min_coeff, max_coeff = _compute_min_max_coeff(min_coeff, max_coeff, frac_strength_treatment)
    # By default, return a plot with 10 points
    # consider 10 values of the effect of the unobserved confounder
    step = (max_coeff - min_coeff) / len_kappa_t
    logger.info("(Min, Max) kappa_t for observed common causes, ({0}, {1})".format(min_coeff, max_coeff))
    if np.equal(max_coeff, min_coeff):
        return max_coeff
    else:
        return np.arange(min_coeff, max_coeff, step)


def _compute_min_max_coeff(min_coeff: float, max_coeff: float, effect_strength_fraction: np.ndarray):
    max_coeff = effect_strength_fraction * max_coeff
    min_coeff = effect_strength_fraction * min_coeff
    return min_coeff, max_coeff


def _infer_default_kappa_y(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    outcome_name: List[str],
    effect_on_y: str,
    frac_strength_outcome: float,
    len_kappa_y: int = 10,
):
    """Infer default effect strength of simulated confounder on treatment."""
    observed_common_causes_names = target_estimand.get_backdoor_variables()
    if len(observed_common_causes_names) > 0:
        observed_common_causes = data[observed_common_causes_names]
        observed_common_causes = pd.get_dummies(observed_common_causes, drop_first=True)
    else:
        raise ValueError(
            "There needs to be at least one common cause to"
            + "automatically compute the default value of kappa_y."
            + " Provide a value for kappa_y"
        )
    y = data[outcome_name]
    # Standardizing the data
    observed_common_causes = StandardScaler().fit_transform(observed_common_causes)
    if effect_on_y == "binary_flip":
        # Fit a model containing all confounders and compare predictions
        # using all features compared to all features except a given
        # confounder.
        ymodel = LogisticRegression().fit(observed_common_causes, y)
        ypred = ymodel.predict(observed_common_causes).astype(int)
        flips = []
        for i in range(observed_common_causes.shape[1]):
            oldval = np.copy(observed_common_causes[:, i])
            observed_common_causes[:, i] = 0
            ycap = ymodel.predict(observed_common_causes).astype(int)
            observed_common_causes[:, i] = oldval
            flips.append(np.sum(abs(ycap - ypred)) / ypred.shape[0])
        min_coeff, max_coeff = min(flips), max(flips)
    elif effect_on_y == "linear":
        corrcoef_var_y = np.corrcoef(observed_common_causes, y, rowvar=False)[-1, :-1]
        std_dev_y = np.std(y)[0]
        max_coeff = max(corrcoef_var_y) * std_dev_y
        min_coeff = min(corrcoef_var_y) * std_dev_y
    else:
        raise NotImplementedError("'" + effect_on_y + "' method not supported for confounders' effect on outcome")
    min_coeff, max_coeff = _compute_min_max_coeff(min_coeff, max_coeff, frac_strength_outcome)
    # By default, return a plot with 10 points
    # consider 10 values of the effect of the unobserved confounder
    step = (max_coeff - min_coeff) / len_kappa_y
    logger.info("(Min, Max) kappa_y for observed common causes, ({0}, {1})".format(min_coeff, max_coeff))
    if np.equal(max_coeff, min_coeff):
        return max_coeff
    else:
        return np.arange(min_coeff, max_coeff, step)


def _include_confounders_effect(
    data: pd.DataFrame,
    new_data: pd.DataFrame,
    effect_on_t: str,
    treatment_name: str,
    kappa_t: float,
    effect_on_y: str,
    outcome_name: str,
    kappa_y: float,
):
    """
    This function deals with the change in the value of the data due to the effect of the unobserved confounder.
    In the case of a binary flip, we flip only if the random number is greater than the threshold set.
    In the case of a linear effect, we use the variable as the linear regression constant.

    :param new_data: pandas.DataFrame: The data to be changed due to the effects of the unobserved confounder.
    :param kappa_t: numpy.float64: The value of the threshold for binary_flip or the value of the regression coefficient for linear effect.
    :param kappa_y: numpy.float64: The value of the threshold for binary_flip or the value of the regression coefficient for linear effect.

    :return: pandas.DataFrame: The DataFrame that includes the effects of the unobserved confounder.
    """
    num_rows = data.shape[0]
    stdnorm = scipy.stats.norm()
    w_random = stdnorm.rvs(num_rows)

    if effect_on_t == "binary_flip":
        alpha = 2 * kappa_t - 1 if kappa_t >= 0.5 else 1 - 2 * kappa_t
        interval = stdnorm.interval(alpha)
        rel_interval = interval[0] if kappa_t >= 0.5 else interval[1]
        new_data.loc[rel_interval <= w_random, treatment_name] = (
            1 - new_data.loc[rel_interval <= w_random, treatment_name]
        )
        for tname in treatment_name:
            if pd.api.types.is_bool_dtype(data[tname]):
                new_data = new_data.astype({tname: "bool"}, copy=False)
    elif effect_on_t == "linear":
        confounder_t_effect = kappa_t * w_random
        # By default, we add the effect of simulated confounder for treatment.
        # But subtract it from outcome to create a negative correlation
        # assuming that the original confounder's effect was positive on both.
        # This is to remove the effect of the original confounder.
        new_data[treatment_name] = new_data[treatment_name].values + np.ndarray(
            shape=(num_rows, 1), buffer=confounder_t_effect
        )
    else:
        raise NotImplementedError("'" + effect_on_t + "' method not supported for confounders' effect on treatment")

    if effect_on_y == "binary_flip":
        alpha = 2 * kappa_y - 1 if kappa_y >= 0.5 else 1 - 2 * kappa_y
        interval = stdnorm.interval(alpha)
        rel_interval = interval[0] if kappa_y >= 0.5 else interval[1]
        new_data.loc[rel_interval <= w_random, outcome_name] = 1 - new_data.loc[rel_interval <= w_random, outcome_name]
        for yname in outcome_name:
            if pd.api.types.is_bool_dtype(data[yname]):
                new_data = new_data.astype({yname: "bool"}, copy=False)
    elif effect_on_y == "linear":
        confounder_y_effect = (-1) * kappa_y * w_random
        # By default, we add the effect of simulated confounder for treatment.
        # But subtract it from outcome to create a negative correlation
        # assuming that the original confounder's effect was positive on both.
        # This is to remove the effect of the original confounder.
        new_data[outcome_name] = new_data[outcome_name].values + np.ndarray(
            shape=(num_rows, 1), buffer=confounder_y_effect
        )
    else:
        raise NotImplementedError("'" + effect_on_y + "' method not supported for confounders' effect on outcome")
    return new_data


def include_simulated_confounder(
    data: pd.DataFrame,
    treatment_name: str,
    outcome_name: str,
    kappa_t: float,
    kappa_y: float,
    variables_of_interest: List,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    c_star_max: int = DEFAULT_C_STAR_MAX,
):
    """
    This function simulates an unobserved confounder based on the data using the following steps:
    1. It calculates the "residuals"  from the treatment and outcome model
    i.) The outcome model has outcome as the dependent variable and all the observed variables including treatment as independent variables
    ii.) The treatment model has treatment as the dependent variable and all the observed variables as independent variables.

    2. U is an intermediate random variable drawn from the normal distribution with the weighted average of residuals as mean and a unit variance
    U ~ N(c1*d_y + c2*d_t, 1)
    where
    *d_y and d_t are residuals from the treatment and outcome model
    *c1 and c2 are coefficients to the residuals

    3. The final U, which is the simulated unobserved confounder is obtained by debiasing the intermediate variable U by residualising it with X


    Choosing the coefficients c1 and c2:
    The coefficients are chosen based on these basic assumptions:
    1. There is a hyperbolic relationship satisfying c1*c2 = c_star
    2. c_star is chosen from a range of possible values based on the correlation of the obtained simulated variable with outcome and treatment.
    3. The product of correlations with treatment and outcome should be at a minimum distance to the maximum correlations with treatment and outcome in any of the observed confounders
    4. The ratio of the weights should be such that they maintain the ratio of the maximum possible observed coefficients within some confidence interval

    :param c_star_max: The maximum possible value for the hyperbolic curve on which the coefficients to the residuals lie. It defaults to 1000 in the code if not specified by the user.
        :type int
    :param convergence_threshold: The threshold to check the plateauing of the correlation while selecting a c_star. It defaults to 0.1 in the code if not specified by the user
        :type float

    :returns: The simulated values of the unobserved confounder based on the data
        :type pandas.core.series.Series

    """

    # Obtaining the list of observed variables
    required_variables = True
    observed_variables = choose_variables(required_variables, variables_of_interest)

    observed_variables_with_treatment_and_outcome = observed_variables + treatment_name + outcome_name

    # Taking a subset of the dataframe that has only observed variables
    data = data[observed_variables_with_treatment_and_outcome]

    # Residuals from the outcome model obtained by fitting a linear model
    y = data[outcome_name[0]]
    observed_variables_with_treatment = observed_variables + treatment_name
    X = data[observed_variables_with_treatment]
    model = sm.OLS(y, X.astype("float"))
    results = model.fit()
    residuals_y = y - results.fittedvalues
    d_y = list(pd.Series(residuals_y))

    # Residuals from the treatment model obtained by fitting a linear model
    t = data[treatment_name[0]].astype("int64")
    X = data[observed_variables]
    model = sm.OLS(t, X)
    results = model.fit()
    residuals_t = t - results.fittedvalues
    d_t = list(pd.Series(residuals_t))

    # Initialising product_cor_metric_observed with a really low value as finding maximum
    product_cor_metric_observed = -10000000000

    for i in observed_variables:
        current_obs_confounder = data[i]
        outcome_values = data[outcome_name[0]]
        correlation_y = current_obs_confounder.corr(outcome_values)
        treatment_values = t
        correlation_t = current_obs_confounder.corr(treatment_values)
        product_cor_metric_current = correlation_y * correlation_t
        if product_cor_metric_current >= product_cor_metric_observed:
            product_cor_metric_observed = product_cor_metric_current
            correlation_t_observed = correlation_t
            correlation_y_observed = correlation_y

    # The user has an option to give the the effect_strength_on_y and effect_strength_on_t which can be then used instead of maximum correlation with treatment and outcome in the observed variables as it specifies the desired effect.
    if kappa_t is not None:
        correlation_t_observed = kappa_t
    if kappa_y is not None:
        correlation_y_observed = kappa_y

    # Choosing a c_star based on the data.
    # The correlations stop increasing upon increasing c_star after a certain value, that is it plateaus and we choose the value of c_star to be the value it plateaus.

    correlation_y_list = []
    correlation_t_list = []
    product_cor_metric_simulated_list = []
    x_list = []

    step = int(c_star_max / 10)
    for i in range(0, int(c_star_max), step):
        c1 = math.sqrt(i)
        c2 = c1
        final_U = _generate_confounder_from_residuals(c1, c2, d_y, d_t, X)
        current_simulated_confounder = final_U
        outcome_values = data[outcome_name[0]]
        correlation_y = current_simulated_confounder.corr(outcome_values)
        correlation_y_list.append(correlation_y)

        treatment_values = t
        correlation_t = current_simulated_confounder.corr(treatment_values)
        correlation_t_list.append(correlation_t)

        product_cor_metric_simulated = correlation_y * correlation_t
        product_cor_metric_simulated_list.append(product_cor_metric_simulated)

        x_list.append(i)

    index = 1
    while index < len(correlation_y_list):
        if (correlation_y_list[index] - correlation_y_list[index - 1]) <= convergence_threshold:
            c_star = x_list[index]
            break
        index = index + 1

    # Choosing c1 and c2 based on the hyperbolic relationship once c_star is chosen by going over various combinations of c1 and c2 values and choosing the combination which
    # which maintains the minimum distance between the product of correlations of the simulated variable and the product of maximum correlations of one of the observed variables
    # and additionally checks if the ratio of the weights are such that they maintain the ratio of the maximum possible observed coefficients within some confidence interval

    # c1_final and c2_final are initialised to the values on the hyperbolic curve such that c1_final = c2_final  and c1_final*c2_final = c_star
    c1_final = math.sqrt(c_star)
    c2_final = math.sqrt(c_star)

    # initialising min_distance_between_product_cor_metrics to be a value greater than 1
    min_distance_between_product_cor_metrics = 1.5
    i = 0.05

    threshold = c_star / 0.05

    while i <= threshold:
        c2 = i
        c1 = c_star / c2
        final_U = _generate_confounder_from_residuals(c1, c2, d_y, d_t, X)

        current_simulated_confounder = final_U
        outcome_values = data[outcome_name[0]]
        correlation_y = current_simulated_confounder.corr(outcome_values)

        treatment_values = t
        correlation_t = current_simulated_confounder.corr(treatment_values)

        product_cor_metric_simulated = correlation_y * correlation_t

        if min_distance_between_product_cor_metrics >= abs(product_cor_metric_simulated - product_cor_metric_observed):
            min_distance_between_product_cor_metrics = abs(product_cor_metric_simulated - product_cor_metric_observed)
            additional_condition = correlation_y_observed / correlation_t_observed
            if ((c1 / c2) <= (additional_condition + 0.3 * additional_condition)) and (
                (c1 / c2) >= (additional_condition - 0.3 * additional_condition)
            ):  # choose minimum positive value
                c1_final = c1
                c2_final = c2

        i = i * 1.5

    """#closed form solution

    print("c_star_max before closed form", c_star_max)

    if max_correlation_with_t == -1000:
        c2 = 0
        c1 = c_star_max
    else:
        additional_condition = abs(max_correlation_with_y/max_correlation_with_t)
        print("additional_condition", additional_condition)
        c2 = math.sqrt(c_star_max/additional_condition)
        c1 = c_star_max/c2"""

    final_U = _generate_confounder_from_residuals(c1_final, c2_final, d_y, d_t, X)

    return final_U


def _generate_confounder_from_residuals(c1, c2, d_y, d_t, X):
    """
    This function takes the residuals from the treatment and outcome model and their coefficients and simulates the intermediate random variable U by taking
    the row wise normal distribution corresponding to each residual value and then debiasing the intermediate variable to get the final variable.

    :param c1: coefficient to the residual from the outcome model
    :type float
    :param c2: coefficient to the residual from the treatment model
    :type float
    :param d_y: residuals from the outcome model
    :type list
    :param d_t: residuals from the treatment model
    :type list

    :returns: The simulated values of the unobserved confounder based on the data
    :type pandas.core.series.Series

    """
    U = []

    for j in range(len(d_t)):
        simulated_variable_mean = c1 * d_y[j] + c2 * d_t[j]
        simulated_variable_stddev = 1
        U.append(np.random.normal(simulated_variable_mean, simulated_variable_stddev, 1))

    U = np.array(U)
    model = sm.OLS(U, X)
    results = model.fit()
    U = U.reshape(
        -1,
    )
    final_U = U - results.fittedvalues.values
    final_U = pd.Series(U)

    return final_U


def sensitivity_linear_partial_r2(
    data: pd.DataFrame,
    estimate: CausalEstimate,
    treatment_name: str,
    frac_strength_treatment: float = 1.0,
    frac_strength_outcome: float = 1.0,
    percent_change_estimate: float = 1.0,
    benchmark_common_causes: Optional[List[str]] = None,
    significance_level: Optional[float] = None,
    null_hypothesis_effect: Optional[float] = None,
    plot_estimate: bool = True,
) -> LinearSensitivityAnalyzer:
    """Add an unobserved confounder for refutation using Linear partial R2 methond (Sensitivity Analysis for linear models).

    :param data: pd.DataFrame: Data to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param treatment_name: str: Name of the treatment
    :param frac_strength_treatment: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on treatment. Defaults to 1.
    :param frac_strength_outcome: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on outcome. Defaults to 1.
    :param percent_change_estimate: It is the percentage of reduction of treatment estimate that could alter the results (default = 1).
                                    if percent_change_estimate = 1, the robustness value describes the strength of association of confounders with treatment and outcome in order to reduce the estimate by 100% i.e bring it down to 0. (relevant only for Linear Sensitivity Analysis, ignore for rest)
    :param benchmark_common_causes: names of variables for bounding strength of confounders. (relevant only for partial-r2 based simulation methods)
    :param significance_level: confidence interval for statistical inference(default = 0.05). (relevant only for partial-r2 based simulation methods)
    :param null_hypothesis_effect: assumed effect under the null hypothesis. (relevant only for linear-partial-R2, ignore for rest)
    :param plot_estimate: Generate contour plot for estimate while performing sensitivity analysis. (default = True).
                            (relevant only for partial-r2 based simulation methods)

    """

    if not (isinstance(estimate.estimator, LinearRegressionEstimator)):
        raise NotImplementedError("Currently only LinearRegressionEstimator is supported for Sensitivity Analysis")

    if len(estimate.estimator._effect_modifier_names) > 0:
        raise NotImplementedError("The current implementation does not support effect modifiers")

    if frac_strength_outcome == 1:
        frac_strength_outcome = frac_strength_treatment

    analyzer = LinearSensitivityAnalyzer(
        estimator=estimate.estimator,
        data=data,
        treatment_name=treatment_name,
        percent_change_estimate=percent_change_estimate,
        significance_level=significance_level,
        benchmark_common_causes=benchmark_common_causes,
        null_hypothesis_effect=null_hypothesis_effect,
        frac_strength_treatment=frac_strength_treatment,
        frac_strength_outcome=frac_strength_outcome,
        common_causes_order=estimate.estimator._observed_common_causes.columns,
    )

    analyzer.check_sensitivity(plot=plot_estimate)
    return analyzer


def sensitivity_non_parametric_partial_r2(
    estimate: CausalEstimate,
    kappa_t: Optional[Union[float, np.ndarray]] = None,
    kappa_y: Optional[Union[float, np.ndarray]] = None,
    frac_strength_treatment: float = 1.0,
    frac_strength_outcome: float = 1.0,
    benchmark_common_causes: Optional[List[str]] = None,
    plot_estimate: bool = True,
    alpha_s_estimator_list: Optional[List] = None,
    alpha_s_estimator_param_list: Optional[List[Dict]] = None,
    g_s_estimator_list: Optional[List] = None,
    g_s_estimator_param_list: Optional[List[Dict]] = None,
    plugin_reisz: bool = False,
) -> Union[PartialLinearSensitivityAnalyzer, NonParametricSensitivityAnalyzer]:
    """Add an unobserved confounder for refutation using Non-parametric partial R2 methond (Sensitivity Analysis for non-parametric models).

    :param estimate: CausalEstimate: Estimate to run the refutation
    :param kappa_t: float, numpy.ndarray: Partial R2 of the unobserved confounder wrt the treatment conditioned on the observed confounders. Only in the case of general non-parametric-partial-R2, it is the fraction of variance in the reisz representer that is explained by the unobserved confounder; specifically  (1-r), where r is the ratio of variance of reisz representer, alpha^2, based on observed confounders and that based on all confounders.
    :param kappa_y: float, numpy.ndarray: Partial R2 of the unobserved confounder wrt the outcome conditioned on the treatment and observed confounders.
    :param frac_strength_treatment: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on treatment. Defaults to 1.
    :param frac_strength_outcome: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on outcome. Defaults to 1.
    :param benchmark_common_causes: names of variables for bounding strength of confounders. (relevant only for partial-r2 based simulation methods)
    :param plot_estimate: Generate contour plot for estimate while performing sensitivity analysis. (default = True).
                            (relevant only for partial-r2 based simulation methods)
    :param alpha_s_estimator_list: list of estimator objects for estimating alpha_s. These objects should have fit() and predict() methods (relevant only for non-parametric-partial-R2 method)
    :param alpha_s_estimator_param_list: list of dictionaries with parameters for finding alpha_s. (relevant only for non-parametric-partial-R2 simulation method)
    :param g_s_estimator_list: list of estimator objects for finding g_s. These objects should have fit() and predict() functions implemented. (relevant only for non-parametric-partial-R2 simulation method)
    :param g_s_estimator_param_list: list of dictionaries with parameters for tuning respective estimators in "g_s_estimator_list". The order of the dictionaries in the list should be consistent with the estimator objects order in "g_s_estimator_list". (relevant only for non-parametric-partial-R2 simulation method)
    :plugin_reisz: bool: Flag on whether to use the plugin estimator or the nonparametric estimator for reisz representer function (alpha_s).
    """

    import dowhy.causal_estimators.econml

    # If the estimator used is LinearDML, partially linear sensitivity analysis will be automatically chosen
    if isinstance(estimate.estimator, dowhy.causal_estimators.econml.Econml):
        if estimate.estimator._econml_methodname == "econml.dml.LinearDML":
            analyzer = PartialLinearSensitivityAnalyzer(
                estimator=estimate._estimator_object,
                observed_common_causes=estimate.estimator._observed_common_causes,
                treatment=estimate.estimator._treatment,
                outcome=estimate.estimator._outcome,
                alpha_s_estimator_param_list=alpha_s_estimator_param_list,
                g_s_estimator_list=g_s_estimator_list,
                g_s_estimator_param_list=g_s_estimator_param_list,
                effect_strength_treatment=kappa_t,
                effect_strength_outcome=kappa_y,
                benchmark_common_causes=benchmark_common_causes,
                frac_strength_treatment=frac_strength_treatment,
                frac_strength_outcome=frac_strength_outcome,
            )
            analyzer.check_sensitivity(plot=plot_estimate)
            return analyzer

    analyzer = NonParametricSensitivityAnalyzer(
        estimator=estimate.estimator,
        observed_common_causes=estimate.estimator._observed_common_causes,
        treatment=estimate.estimator._treatment,
        outcome=estimate.estimator._outcome,
        alpha_s_estimator_list=alpha_s_estimator_list,
        alpha_s_estimator_param_list=alpha_s_estimator_param_list,
        g_s_estimator_list=g_s_estimator_list,
        g_s_estimator_param_list=g_s_estimator_param_list,
        effect_strength_treatment=kappa_t,
        effect_strength_outcome=kappa_y,
        benchmark_common_causes=benchmark_common_causes,
        frac_strength_treatment=frac_strength_treatment,
        frac_strength_outcome=frac_strength_outcome,
        theta_s=estimate.value,
        plugin_reisz=plugin_reisz,
    )
    analyzer.check_sensitivity(plot=plot_estimate)
    return analyzer


def sensitivity_e_value(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    treatment_name: List[str],
    outcome_name: List[str],
    plot_estimate: bool = True,
) -> EValueSensitivityAnalyzer:
    if not isinstance(estimate.estimator, RegressionEstimator):
        raise NotImplementedError("E-Value sensitivity analysis is currently only implemented RegressionEstimator.")

    if len(estimate.estimator._effect_modifier_names) > 0:
        raise NotImplementedError("The current implementation does not support effect modifiers")

    analyzer = EValueSensitivityAnalyzer(
        estimate=estimate,
        estimand=target_estimand,
        data=data,
        treatment_name=treatment_name[0],
        outcome_name=outcome_name[0],
    )
    analyzer.check_sensitivity(plot=plot_estimate)
    return analyzer


def sensitivity_simulation(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    treatment_name: str,
    outcome_name: str,
    kappa_t: Optional[Union[float, np.ndarray]] = None,
    kappa_y: Optional[Union[float, np.ndarray]] = None,
    confounders_effect_on_treatment: str = "binary_flip",
    confounders_effect_on_outcome: str = "linear",
    frac_strength_treatment: float = 1.0,
    frac_strength_outcome: float = 1.0,
    plotmethod: Optional[str] = None,
    show_progress_bar=False,
    **_,
) -> CausalRefutation:
    """
    This function attempts to add an unobserved common cause to the outcome and the treatment. At present, we have implemented the behavior for one dimensional behaviors for continuous
    and binary variables. This function can either take single valued inputs or a range of inputs. The function then looks at the data type of the input and then decides on the course of
    action.

    :param data: pd.DataFrame: Data to run the refutation
    :param target_estimand: IdentifiedEstimand: Identified estimand to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param treatment_name: str: Name of the treatment
    :param outcome_name: str: Name of the outcome
    :param kappa_t: float, numpy.ndarray: Strength of the confounder's effect on treatment. When confounders_effect_on_treatment is linear,  it is the regression coefficient. When the confounders_effect_on_treatment is binary flip, it is the probability with which effect of unobserved confounder can invert the value of the treatment.
    :param kappa_y: float, numpy.ndarray: Strength of the confounder's effect on outcome. Its interpretation depends on confounders_effect_on_outcome and the simulation_method. When simulation_method is direct-simulation, for a linear effect it behaves like the regression coefficient and for a binary flip, it is the probability with which it can invert the value of the outcome.
    :param confounders_effect_on_treatment: str : The type of effect on the treatment due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
    :param confounders_effect_on_outcome: str : The type of effect on the outcome due to the unobserved confounder. Possible values are ['binary_flip', 'linear']
    :param frac_strength_treatment: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on treatment. Defaults to 1.
    :param frac_strength_outcome: float: This parameter decides the effect strength of the simulated confounder as a fraction of the effect strength of observed confounders on outcome. Defaults to 1.
    :param plotmethod: string: Type of plot to be shown. If None, no plot is generated. This parameter is used only only when more than one treatment confounder effect values or outcome confounder effect values are provided. Default is "colormesh". Supported values are "contour", "colormesh" when more than one value is provided for both confounder effect value parameters; "line" when provided for only one of them.

    :return: CausalRefuter: An object that contains the estimated effect and a new effect and the name of the refutation used.
    """
    if kappa_t is None:
        kappa_t = _infer_default_kappa_t(
            data, target_estimand, treatment_name, confounders_effect_on_treatment, frac_strength_treatment
        )
    if kappa_y is None:
        kappa_y = _infer_default_kappa_y(
            data, target_estimand, outcome_name, confounders_effect_on_outcome, frac_strength_outcome
        )

    if not isinstance(kappa_t, (list, np.ndarray)) and not isinstance(
        kappa_y, (list, np.ndarray)
    ):  # Deal with single value inputs
        new_data = copy.deepcopy(data)
        new_data = _include_confounders_effect(
            data,
            new_data,
            confounders_effect_on_treatment,
            treatment_name,
            kappa_t,
            confounders_effect_on_outcome,
            outcome_name,
            kappa_y,
        )
        new_estimator = CausalEstimator.get_estimator_object(new_data, target_estimand, estimate)
        new_effect = new_estimator.estimate_effect()
        refute = CausalRefutation(
            estimate.value, new_effect.value, refutation_type="Refute: Add an Unobserved Common Cause"
        )

        refute.new_effect_array = np.array(new_effect.value)
        refute.new_effect = new_effect.value
        return refute

    else:  # Deal with multiple value inputs

        if isinstance(kappa_t, (list, np.ndarray)) and isinstance(
            kappa_y, (list, np.ndarray)
        ):  # Deal with range inputs
            # Get a 2D matrix of values
            # x,y =  np.meshgrid(self.kappa_t, self.kappa_y) # x,y are both MxN

            results_matrix = np.random.rand(len(kappa_t), len(kappa_y))  # Matrix to hold all the results of NxM
            orig_data = copy.deepcopy(data)

            for i in tqdm(
                range(len(kappa_t)),
                colour=CausalRefuter.PROGRESS_BAR_COLOR,
                disable=not show_progress_bar,
                desc="Refuting Estimates: ",
            ):
                for j in range(len(kappa_y)):
                    new_data = _include_confounders_effect(
                        data,
                        orig_data,
                        confounders_effect_on_treatment,
                        treatment_name,
                        kappa_t[i],
                        confounders_effect_on_outcome,
                        outcome_name,
                        kappa_y[j],
                    )
                    new_estimator = CausalEstimator.get_estimator_object(new_data, target_estimand, estimate)
                    new_effect = new_estimator.estimate_effect()
                    refute = CausalRefutation(
                        estimate.value,
                        new_effect.value,
                        refutation_type="Refute: Add an Unobserved Common Cause",
                    )
                    results_matrix[i][j] = refute.new_effect  # Populate the results

            refute.new_effect_array = results_matrix
            refute.new_effect = (np.min(results_matrix), np.max(results_matrix))
            # Store the values into the refute object
            if plotmethod is None:
                return refute

            import matplotlib
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])

            oe = estimate.value
            contour_levels = [oe / 4.0, oe / 2.0, (3.0 / 4) * oe, oe]
            contour_levels.extend([0, np.min(results_matrix), np.max(results_matrix)])
            if plotmethod == "contour":
                cp = plt.contourf(kappa_y, kappa_t, results_matrix, levels=sorted(contour_levels))
                # Adding a label on the contour line for the original estimate
                fmt = {}
                trueeffect_index = np.where(cp.levels == oe)[0][0]
                fmt[cp.levels[trueeffect_index]] = "Estimated Effect"
                # Label every other level using strings
                plt.clabel(cp, [cp.levels[trueeffect_index]], inline=True, fmt=fmt)
                plt.colorbar(cp)
            elif plotmethod == "colormesh":
                cp = plt.pcolormesh(kappa_y, kappa_t, results_matrix, shading="nearest")
                plt.colorbar(cp, ticks=contour_levels)
            ax.yaxis.set_ticks(kappa_t)
            ax.xaxis.set_ticks(kappa_y)
            plt.xticks(rotation=45)
            ax.set_title("Effect of Unobserved Common Cause")
            ax.set_ylabel("Value of Linear Constant on Treatment")
            ax.set_xlabel("Value of Linear Constant on Outcome")
            plt.show()

            return refute

        elif isinstance(kappa_t, (list, np.ndarray)):
            outcomes = np.random.rand(len(kappa_t))
            orig_data = copy.deepcopy(data)

            for i in tqdm(
                range(0, len(kappa_t)),
                colour=CausalRefuter.PROGRESS_BAR_COLOR,
                disable=not show_progress_bar,
                desc="Refuting Estimates: ",
            ):
                new_data = _include_confounders_effect(
                    data,
                    orig_data,
                    confounders_effect_on_treatment,
                    treatment_name,
                    kappa_t[i],
                    confounders_effect_on_outcome,
                    outcome_name,
                    kappa_y,
                )
                new_estimator = CausalEstimator.get_estimator_object(new_data, target_estimand, estimate)
                new_effect = new_estimator.estimate_effect()
                refute = CausalRefutation(
                    estimate.value, new_effect.value, refutation_type="Refute: Add an Unobserved Common Cause"
                )
                logger.debug(refute)
                outcomes[i] = refute.new_effect  # Populate the results

            refute.new_effect_array = outcomes
            refute.new_effect = (np.min(outcomes), np.max(outcomes))
            if plotmethod is None:
                return refute

            import matplotlib
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])

            plt.plot(kappa_t, outcomes)
            plt.axhline(estimate.value, linestyle="--", color="gray")
            ax.set_title("Effect of Unobserved Common Cause")
            ax.set_xlabel("Value of Linear Constant on Treatment")
            ax.set_ylabel("Estimated Effect after adding the common cause")
            plt.show()

            return refute

        elif isinstance(kappa_y, (list, np.ndarray)):
            outcomes = np.random.rand(len(kappa_y))
            orig_data = copy.deepcopy(data)

            for i in tqdm(
                range(0, len(kappa_y)),
                colour=CausalRefuter.PROGRESS_BAR_COLOR,
                disable=not show_progress_bar,
                desc="Refuting Estimates: ",
            ):
                new_data = _include_confounders_effect(
                    data,
                    orig_data,
                    confounders_effect_on_treatment,
                    treatment_name,
                    kappa_t,
                    confounders_effect_on_outcome,
                    outcome_name,
                    kappa_y[i],
                )
                new_estimator = CausalEstimator.get_estimator_object(new_data, target_estimand, estimate)
                new_effect = new_estimator.estimate_effect()
                refute = CausalRefutation(
                    estimate.value, new_effect.value, refutation_type="Refute: Add an Unobserved Common Cause"
                )
                logger.debug(refute)
                outcomes[i] = refute.new_effect  # Populate the results

            refute.new_effect_array = outcomes
            refute.new_effect = (np.min(outcomes), np.max(outcomes))
            if plotmethod is None:
                return refute

            import matplotlib
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5))
            left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
            ax = fig.add_axes([left, bottom, width, height])

            plt.plot(kappa_y, outcomes)
            plt.axhline(estimate.value, linestyle="--", color="gray")
            ax.set_title("Effect of Unobserved Common Cause")
            ax.set_xlabel("Value of Linear Constant on Outcome")
            ax.set_ylabel("Estimated Effect after adding the common cause")
            plt.show()

            return refute
