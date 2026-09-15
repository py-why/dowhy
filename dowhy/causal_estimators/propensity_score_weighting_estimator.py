from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class PropensityScoreWeightingEstimator(PropensityScoreEstimator):
    """Estimate effect of treatment by weighing the data by
    inverse probability of occurrence.

    Straightforward application of the back-door criterion.

    Supports additional parameters as listed below.

    **References**

    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

    Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient estimation of average treatment effects using the estimated propensity score. *Econometrica*, 71(4), 1161–1189. https://doi.org/10.1111/1468-0262.00442

    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        min_ps_score: float = 0.05,
        max_ps_score: float = 0.95,
        weighting_scheme: str = "ips_weight",
        propensity_score_model: Optional[Any] = None,
        propensity_score_column: str = "propensity_score",
        **kwargs,
    ):
        """
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
        :param min_ps_score: Lower bound used to clip the propensity score.
            Default=0.05
        :param max_ps_score: Upper bound used to clip the propensity score.
            Default=0.95
        :param weighting_scheme: Weighting method to use. Can be inverse
            propensity score ("ips_weight", default), stabilized IPS score
            ("ips_stabilized_weight"), or normalized IPS score
            ("ips_normalized_weight").
        :param propensity_score_model: The model used to compute propensity
            score. Can be any classification model that supports fit() and
            predict_proba() methods. If None, use LogisticRegression model as
            the default. Default=None
        :param propensity_score_column: Column name that stores the
            propensity score. Default='propensity_score'
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
            propensity_score_model=propensity_score_model,
            propensity_score_column=propensity_score_column,
            min_ps_score=min_ps_score,
            max_ps_score=max_ps_score,
            weighting_scheme=weighting_scheme,
            **kwargs,
        )

        self.logger.info("INFO: Using Propensity Score Weighting Estimator")

        # Setting method specific parameters
        self.weighting_scheme = weighting_scheme
        self.min_ps_score = min_ps_score
        self.max_ps_score = max_ps_score

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
        super().fit(data, effect_modifier_names=effect_modifier_names)

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        **_,
    ):
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        if self.propensity_score_column not in data:
            self.estimate_propensity_score_column(data)

        # trim propensity score weights
        data[self.propensity_score_column] = np.minimum(self.max_ps_score, data[self.propensity_score_column])
        data[self.propensity_score_column] = np.maximum(self.min_ps_score, data[self.propensity_score_column])

        # ips ==> (isTreated(y)/ps(y)) + ((1-isTreated(y))/(1-ps(y)))
        # nips ==> ips / (sum of ips over all units)
        # icps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all control units)
        # itps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all treatment units)
        ipst_sum = sum(data[self._target_estimand.treatment_variable[0]] / data[self.propensity_score_column])
        ipsc_sum = sum(
            (1 - data[self._target_estimand.treatment_variable[0]]) / (1 - data[self.propensity_score_column])
        )
        num_units = len(data[self._target_estimand.treatment_variable[0]])
        num_treatment_units = sum(data[self._target_estimand.treatment_variable[0]])
        num_control_units = num_units - num_treatment_units

        # Vanilla IPS estimator
        data["ips_weight"] = data[self._target_estimand.treatment_variable[0]] / data[self.propensity_score_column] + (
            1 - data[self._target_estimand.treatment_variable[0]]
        ) / (1 - data[self.propensity_score_column])
        data["tips_weight"] = data[self._target_estimand.treatment_variable[0]] + (
            1 - data[self._target_estimand.treatment_variable[0]]
        ) * data[self.propensity_score_column] / (1 - data[self.propensity_score_column])
        data["cips_weight"] = data[self._target_estimand.treatment_variable[0]] * (
            1 - data[self.propensity_score_column]
        ) / data[self.propensity_score_column] + (1 - data[self._target_estimand.treatment_variable[0]])

        # The Hajek estimator (or the self-normalized estimator)
        data["ips_normalized_weight"] = (
            data[self._target_estimand.treatment_variable[0]] / data[self.propensity_score_column] / ipst_sum
            + (1 - data[self._target_estimand.treatment_variable[0]])
            / (1 - data[self.propensity_score_column])
            / ipsc_sum
        )
        ipst_for_att_sum = sum(data[self._target_estimand.treatment_variable[0]])
        ipsc_for_att_sum = sum(
            (1 - data[self._target_estimand.treatment_variable[0]])
            / (1 - data[self.propensity_score_column])
            * data[self.propensity_score_column]
        )
        data["tips_normalized_weight"] = (
            data[self._target_estimand.treatment_variable[0]] / ipst_for_att_sum
            + (1 - data[self._target_estimand.treatment_variable[0]])
            * data[self.propensity_score_column]
            / (1 - data[self.propensity_score_column])
            / ipsc_for_att_sum
        )
        ipst_for_atc_sum = sum(
            data[self._target_estimand.treatment_variable[0]]
            / data[self.propensity_score_column]
            * (1 - data[self.propensity_score_column])
        )
        ipsc_for_atc_sum = sum((1 - data[self._target_estimand.treatment_variable[0]]))
        data["cips_normalized_weight"] = (
            data[self._target_estimand.treatment_variable[0]]
            * (1 - data[self.propensity_score_column])
            / data[self.propensity_score_column]
            / ipst_for_atc_sum
            + (1 - data[self._target_estimand.treatment_variable[0]]) / ipsc_for_atc_sum
        )

        # Stabilized weights (from Robins, Hernan, Brumback (2000))
        # Paper: Marginal Structural Models and Causal Inference in Epidemiology
        p_treatment = sum(data[self._target_estimand.treatment_variable[0]]) / num_units
        data["ips_stabilized_weight"] = data[self._target_estimand.treatment_variable[0]] / data[
            self.propensity_score_column
        ] * p_treatment + (1 - data[self._target_estimand.treatment_variable[0]]) / (
            1 - data[self.propensity_score_column]
        ) * (
            1 - p_treatment
        )
        data["tips_stabilized_weight"] = data[self._target_estimand.treatment_variable[0]] * p_treatment + (
            1 - data[self._target_estimand.treatment_variable[0]]
        ) * data[self.propensity_score_column] / (1 - data[self.propensity_score_column]) * (1 - p_treatment)
        data["cips_stabilized_weight"] = data[self._target_estimand.treatment_variable[0]] * (
            1 - data[self.propensity_score_column]
        ) / data[self.propensity_score_column] * p_treatment + (
            1 - data[self._target_estimand.treatment_variable[0]]
        ) * (
            1 - p_treatment
        )

        if isinstance(target_units, pd.DataFrame) or target_units == "ate":
            weighting_scheme_name = self.weighting_scheme
        elif target_units == "att":
            weighting_scheme_name = "t" + self.weighting_scheme
        elif target_units == "atc":
            weighting_scheme_name = "c" + self.weighting_scheme
        else:
            raise ValueError(f"Target units value {target_units} not supported")

        # Calculating the effect
        data["d_y"] = (
            data[weighting_scheme_name]
            * data[self._target_estimand.treatment_variable[0]]
            * data[self._target_estimand.outcome_variable[0]]
        )
        data["dbar_y"] = (
            data[weighting_scheme_name]
            * (1 - data[self._target_estimand.treatment_variable[0]])
            * data[self._target_estimand.outcome_variable[0]]
        )
        sum_dy_weights = np.sum(data[self._target_estimand.treatment_variable[0]] * data[weighting_scheme_name])
        sum_dbary_weights = np.sum(
            (1 - data[self._target_estimand.treatment_variable[0]]) * data[weighting_scheme_name]
        )
        # Subtracting the weighted means
        est = data["d_y"].sum() / sum_dy_weights - data["dbar_y"].sum() / sum_dbary_weights

        # Store the arrays needed for influence-function variance estimation.
        # We keep numpy arrays rather than references to the dataframe to avoid
        # issues if the caller mutates the frame after this call.
        self._ipw_T = data[self._target_estimand.treatment_variable[0]].to_numpy().astype(float)
        self._ipw_Y = data[self._target_estimand.outcome_variable[0]].to_numpy().astype(float)
        self._ipw_ps = data[self.propensity_score_column].to_numpy().astype(float)
        self._ipw_estimate = est
        self._ipw_target_units = target_units
        self._ipw_weighting_scheme = self.weighting_scheme

        # TODO - how can we add additional information into the returned estimate?
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._target_estimand.treatment_variable,
            outcome_name=self._target_estimand.outcome_variable,
            estimate=est,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            propensity_scores=data[self.propensity_score_column],
        )

        estimate.add_estimator(self)
        return estimate

    def _compute_influence_function(self):
        """Compute per-unit influence function values for the IPW estimator.

        The influence function (IF) gives the first-order contribution of each
        observation to the estimator. Its sample variance divided by n is a
        consistent estimator of the asymptotic variance of the IPW ATE/ATT/ATC.

        All weighting schemes in this estimator produce a ratio (Hajek-type)
        estimator because estimate_effect divides by the sum of weights in each
        arm. Even "ips_weight" ends up computing:
            sum(T*Y/e) / sum(T/e) - sum((1-T)*Y/(1-e)) / sum((1-T)/(1-e))
        which is the Hajek weighted mean, not the Horvitz-Thompson sum.
        The influence function must account for this ratio structure.

        Reference: Lunceford & Davidian (2004), "Stratification and weighting
        via the propensity score in estimation of causal treatment effects:
        a comparative study", Statistics in Medicine 23(19), pp. 2937-2960.
        """
        T = self._ipw_T
        Y = self._ipw_Y
        e = self._ipw_ps
        n = len(T)
        target = self._ipw_target_units

        if target is None or target == "ate" or isinstance(target, pd.DataFrame):
            # ATE: all weighting schemes here are ratio estimators (Hajek form).
            # ATE = sum(T*Y*w1) / sum(T*w1) - sum((1-T)*Y*w0) / sum((1-T)*w0)
            # The IF for a ratio A/B is (1/B)*(a_i - (A/B)*b_i).
            #
            # When target_units is a DataFrame we use the population ATE IF
            # over all observations. Per-unit custom IF is not supported yet.
            sum_w1 = np.sum(T / e)
            sum_w0 = np.sum((1 - T) / (1 - e))
            mu1 = np.sum(T * Y / e) / sum_w1
            mu0 = np.sum((1 - T) * Y / (1 - e)) / sum_w0
            # Linearized IF for the difference of two ratio estimators
            if_treated = (T / e) * (Y - mu1) / (sum_w1 / n)
            if_control = ((1 - T) / (1 - e)) * (Y - mu0) / (sum_w0 / n)
            influence = if_treated - if_control

        elif target == "att":
            # ATT: E[Y1-Y0 | T=1]
            # Treated arm is the sample mean of Y among treated.
            # Control arm uses weights e/(1-e) to reweight controls.
            # Both arms are ratio estimators, each with its own denominator.
            p = np.mean(T)
            sum_w0_att = np.sum((1 - T) * e / (1 - e))
            mu1_att = np.sum(T * Y) / np.sum(T)
            mu0_att = np.sum((1 - T) * e / (1 - e) * Y) / sum_w0_att
            # Treated arm denominator is mean(T) = p.
            # Control arm denominator is mean((1-T)*e/(1-e)), not p.
            if_treated = T * (Y - mu1_att) / p
            if_control = (1 - T) * e / (1 - e) * (Y - mu0_att) / (sum_w0_att / n)
            influence = if_treated - if_control

        elif target == "atc":
            # ATC: E[Y1-Y0 | T=0]
            # Treated arm uses weights (1-e)/e to reweight treated.
            # Control arm is the sample mean of Y among controls.
            p0 = np.mean(1 - T)
            sum_w1_atc = np.sum(T * (1 - e) / e)
            mu1_atc = np.sum(T * (1 - e) / e * Y) / sum_w1_atc
            mu0_atc = np.sum((1 - T) * Y) / np.sum(1 - T)
            # Treated arm denominator is mean(T*(1-e)/e), not p0.
            # Control arm denominator is mean(1-T) = p0.
            if_treated = T * (1 - e) / e * (Y - mu1_atc) / (sum_w1_atc / n)
            if_control = (1 - T) * (Y - mu0_atc) / p0
            influence = if_treated - if_control

        else:
            raise ValueError(f"Cannot compute analytic SE for target_units={target}")

        return influence

    def _estimate_std_error(self, method=None, **kwargs):
        """Analytic standard error via the influence function.

        The variance of the IPW estimator is consistently estimated by
        sum(IF^2)/n^2, where IF is the per-unit influence function. This
        avoids the computational cost of bootstrapping while giving the
        same large-sample coverage.
        """
        if not hasattr(self, "_ipw_T"):
            raise RuntimeError(
                "estimate_effect() must be called before _estimate_std_error(). "
                "Call model.estimate_effect(..., confidence_intervals=True) instead."
            )
        influence = self._compute_influence_function()
        n = len(influence)
        # We use sum(IF^2)/n^2 rather than var(IF)/n. These are equivalent
        # asymptotically because the IF has mean zero by construction.
        # See Lunceford & Davidian (2004), section 2.3.
        var_hat = np.sum(influence ** 2) / (n ** 2)
        return np.sqrt(var_hat)

    def _estimate_confidence_intervals(self, confidence_level=None, method=None, **kwargs):
        """Confidence interval based on the influence-function SE and
        normal approximation (Wald interval).

        This is the standard large-sample interval for IPW estimators.
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        se = self._estimate_std_error()
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ate = self._ipw_estimate
        return (ate - z * se, ate + z * se)

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_adjustment_set()
        expr += "+".join(var_list)
        return expr
