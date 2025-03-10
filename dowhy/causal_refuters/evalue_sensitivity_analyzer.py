import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_estimators.generalized_linear_model_estimator import GeneralizedLinearModelEstimator
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class EValueSensitivityAnalyzer:
    """
    This class computes Ding & VanderWeele's E-value for unmeasured confounding. The E-value is the minimum
    strength of association on the risk ratio scale that an unmeasured confounder would need to have with
    both the treatment and the outcome, conditional on the measured covariates, to fully explain away a specific
    treatment-outcome association.

    It benchmarks the E-value against measured confounders using McGowan and Greevy Jr.'s Observed Covariate E-value.
    This approach drops measured confounders and re-fits the estimator, measuring how much the limiting bound of the
    confidence interval changes on the E-value scale. This benchmarks hypothetical unmeasured confounding against each
    of the measured confounders.

    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4820664/,
    https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf, and
    https://arxiv.org/pdf/2011.07030.pdf.
    The implementation is based on the R packages https://github.com/cran/EValue and https://github.com/LucyMcGowan/tipr.

    :param estimate: CausalEstimate
    :param estimand: IdentifiedEstimand
    :param data: pd.DataFrame
    :param outcome_name: Outcome variable name
    :param no_effect_baseline: A number to which to shift the observed estimate to. Defaults to 1 for ratio measures (RR, OR, HR)
    and 0 for additive measures (OLS, MD). (Default = None)
    """

    def __init__(
        self,
        estimate: CausalEstimate,
        estimand: IdentifiedEstimand,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        no_effect_baseline=None,
    ):

        self.estimate = estimate
        self.estimand = estimand
        self.data = data
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.no_effect_baseline = no_effect_baseline
        if self.no_effect_baseline is None:
            if isinstance(self.estimate.estimator, LinearRegressionEstimator):
                self.no_effect_baseline = 0
            elif isinstance(self.estimate.estimator, GeneralizedLinearModelEstimator):
                self.no_effect_baseline = 1
        self.logger = logging.getLogger(__name__)
        self.stats = None
        self.benchmarking_results = None
        self.sd_outcome = np.std(self.data[outcome_name])

    def check_sensitivity(self, data: pd.DataFrame, plot=True):
        """
        Computes E-value for point estimate and confidence limits. Benchmarks E-values against
        measured confounders using Observed Covariate E-values. Plots E-values and Observed
        Covariate E-values.

        :param plot: plots E-value for point estimate and confidence limit. (Default = True)
        """

        if isinstance(self.estimate.estimator, LinearRegressionEstimator):
            coef_est = self.estimate.value
            coef_se = self.estimate.get_standard_error()[0]
        elif isinstance(self.estimate.estimator, GeneralizedLinearModelEstimator):
            coef_est = self.estimate.estimator.model.params[1:2].to_numpy()[0]
            coef_se = self.estimate.estimator.model.bse[1:2].to_numpy()[0]
        self.stats = self.get_evalue(coef_est, coef_se)

        eval_lo = self.stats["evalue_lower_ci"]
        eval_hi = self.stats["evalue_upper_ci"]
        if (eval_lo is None and eval_hi == 1) or (eval_hi is None and eval_lo == 1):
            self.logger.info(
                "Not benchmarking with Observed Covariate E-values. Confidence interval is already tipped."
            )
        else:
            self.benchmark(data)

        if plot:
            self.plot()

    def get_evalue(self, coef_est, coef_se):
        """
        Computes E-value for point estimate and confidence limits. The estimate and
        confidence limits are converted to the risk ratio scale before the E-value is calculated.

        :param coef_est: coefficient estimate
        :param coef_se: coefficient standard error
        """

        if isinstance(self.estimate.estimator, LinearRegressionEstimator):
            return self._evalue_OLS(coef_est, coef_se, self.sd_outcome, self.no_effect_baseline)
        elif isinstance(self.estimate.estimator, GeneralizedLinearModelEstimator):
            est = np.exp(coef_est)
            lo = np.exp(coef_est - 1.96 * coef_se)
            hi = np.exp(coef_est + 1.96 * coef_se)

            family = self.estimate.estimator.family
            link = self.estimate.estimator.family.link
            if isinstance(family, sm.families.Binomial) and isinstance(link, sm.families.links.Logit):
                rare = self.data[self.outcome_name].mean() < 0.15
                return self._evalue_OR(est, lo, hi, rare, self.no_effect_baseline)
            elif isinstance(
                family,
                (sm.families.Poisson, sm.families.NegativeBinomial, sm.families.Gamma),
            ) and isinstance(link, sm.families.links.Log):
                return self._evalue_RR(est, lo, hi, self.no_effect_baseline)
            else:
                raise NotImplementedError(
                    "Currently, only four GLM families are supported: "
                    "1) sm.families.Binomial(link=sm.families.links.logit), "
                    "2) sm.families.Poisson(link=sm.families.links.log), "
                    "3) sm.families.NegativeBinomial(link=sm.families.links.log), "
                    "4) sm.families.Gamma(link=sm.families.links.log), "
                )

    def plot(
        self,
        num_points_per_contour=200,
        plot_size=(6.4, 4.8),
        contour_colors=["blue", "red"],
        benchmarking_color="green",
        xy_limit=None,
    ):
        """
        Plots contours showing the combinations of treatment-confounder and confounder-outcome
        risk ratios that would tip the point estimate and confidence limit. The X-axis shows
        the treatment-confounder risk ratio and the Y-axis shows the confounder-outcome risk ratio.

        :param num_points_per_contour: number of points to calculate and plot for each contour (Default = 200)
        :param plot_size: size of the plot (Default = (6.4,4.8))
        :param contour_colors: colors for point estimate and confidence limit contour (Default = ["blue", "red"])
        :param benchmarking_color: color for observed covariate E-values. (Default = "green")
        :param xy_limit: plot's maximum x and y value. Default is 2 x E-value. (Default = None)
        """

        if self.stats is None:
            raise ValueError("Must call ``check_sensitivity'' before calling ``plot''")

        fig, ax = plt.subplots(1, 1, figsize=plot_size)

        eval_est = self.stats["evalue_estimate"]
        rr = self.stats["converted_estimate"]
        if rr < 1:
            rr = 1 / rr

        if xy_limit is None:
            xy_limit = eval_est * 2

        self._plot_contour(ax, rr, eval_est, num_points_per_contour, contour_colors[0], xy_limit)

        eval_lo = self.stats["evalue_lower_ci"]
        eval_hi = self.stats["evalue_upper_ci"]
        if (eval_lo is None and eval_hi == 1) or (eval_hi is None and eval_lo == 1):
            self.logger.info("Plotting contour for point estimate only. Confidence interval is already tipped.")
        else:
            if eval_lo is None:
                rr_ci = self.stats["converted_upper_ci"]
                eval_ci = eval_hi
            else:
                rr_ci = self.stats["converted_lower_ci"]
                eval_ci = eval_lo
            if rr_ci < 1:
                rr_ci = 1 / rr_ci

            self._plot_contour(
                ax,
                rr_ci,
                eval_ci,
                num_points_per_contour,
                contour_colors[1],
                xy_limit,
                point_est=False,
            )

            ax.scatter(
                self.benchmarking_results["observed_covariate_e_value"],
                self.benchmarking_results["observed_covariate_e_value"],
                label="Observed Covariate E-values",
                color=benchmarking_color,
            )
            example_var = self.benchmarking_results.iloc[0]
            obs_evalue = example_var["observed_covariate_e_value"]
            ax.text(obs_evalue, obs_evalue, example_var.name)

        ax.set(xlabel="$RR_{treatment-confounder}$", ylabel="$RR_{confounder-outcome}$")
        plt.ylim(1, xy_limit)
        plt.xlim(1, xy_limit)
        plt.legend()
        plt.show()

    def _plot_contour(self, ax, rr, evalue, n_pts, color, xy_limit, point_est=True):
        """
        Plots a single contour line

        :param ax: matplotlib axis
        :param rr: observed point estimate/confidence limit on the risk ratio scale
        :param evalue: E-value corresponding to observed point estimate/confidence limit
        :param n_pts: number of points to calculate and plot for each contour
        :param color: color for contour
        :param xy_limit: plot's maximum x and y value
        :param point_est: whether this is the point estimate (Default = True)
        """

        step = (xy_limit - rr) / n_pts
        x_est = np.linspace(rr + step, xy_limit, num=n_pts)
        y_est = rr * (rr - 1) / (x_est - rr) + rr

        est_string = "point estimate" if point_est else "confidence interval"
        ax.scatter(
            evalue,
            evalue,
            label=f"E-value for {est_string}: {evalue.round(2)}",
            color=color,
        )
        ax.fill_between(
            x_est,
            y_est,
            xy_limit,
            color=color,
            alpha=0.2,
            label=f"Tips {est_string}",
        )
        ax.plot(x_est, y_est, color=color)

    def benchmark(self, data: pd.DataFrame):
        """
        Benchmarks E-values against the measured confounders using McGowan and Greevy Jr.'s Observed
        Covariate E-value. This approach drops measured confounders and re-fits the estimator, measuring
        how much the limiting bound of the confidence interval changes on the E-value scale. This benchmarks
        hypothetical unmeasured confounding against each of the measured confounders.

        See: https://arxiv.org/pdf/2011.07030.pdf and https://github.com/LucyMcGowan/tipr
        """

        new_ests = []
        new_lo = []
        new_hi = []
        observed_covariate_e_values = []
        covariates = self.estimand.get_adjustment_set()
        for drop_var in covariates:

            # new estimator
            new_covariate_vars = [var for var in covariates if var != drop_var]
            new_estimand = copy.deepcopy(self.estimand)
            new_estimand.set_adjustment_set(new_covariate_vars)
            new_estimator = self.estimate.estimator.get_new_estimator_object(new_estimand)
            new_estimator.fit(
                self.data,
                effect_modifier_names=self.estimate.estimator._effect_modifier_names,
                **new_estimator._fit_params if hasattr(new_estimator, "_fit_params") else {},
            )

            # new effect estimate
            new_effect = new_estimator.estimate_effect(
                self.data,
                control_value=self.estimate.control_value,
                treatment_value=self.estimate.treatment_value,
                target_units=self.estimate.estimator._target_units,
            )
            if isinstance(self.estimate.estimator, LinearRegressionEstimator):
                coef_est = new_effect.value
                coef_se = new_effect.get_standard_error()[0]
            elif isinstance(self.estimate.estimator, GeneralizedLinearModelEstimator):
                coef_est = new_effect.estimator.model.params[1:2].to_numpy()[0]
                coef_se = new_effect.estimator.model.bse[1:2].to_numpy()[0]
            new_stats = self.get_evalue(coef_est, coef_se)

            # observed covariate E-value
            if self.stats["evalue_lower_ci"] is None:
                ci = self.stats["converted_upper_ci"]
                new_ci = new_stats["converted_upper_ci"]
            else:
                ci = self.stats["converted_lower_ci"]
                new_ci = new_stats["converted_lower_ci"]
            covariate_e_value = self._observed_covariate_e_value(ci, new_ci)

            new_ests.append(new_stats["converted_estimate"])
            new_lo.append(new_stats["converted_lower_ci"])
            new_hi.append(new_stats["converted_upper_ci"])
            observed_covariate_e_values.append(covariate_e_value)

        self.benchmarking_results = pd.DataFrame(
            {
                "dropped_covariate": covariates,
                "converted_est": new_ests,
                "converted_lower_ci": new_lo,
                "converted_upper_ci": new_hi,
                "observed_covariate_e_value": observed_covariate_e_values,
            }
        ).sort_values(by="observed_covariate_e_value", ascending=False)
        self.benchmarking_results = self.benchmarking_results.set_index("dropped_covariate")

    def _observed_covariate_e_value(self, ci, new_ci):
        """
        Computes Observed Covariate E-value given effect estimate from new model without
        a specific measured confounder.

        Based on: https://github.com/LucyMcGowan/tipr/blob/master/R/observed_covariate_e_value.R

        :param ci: limiting confidence bound from original model on risk ratio scale
        :param new_ci: limiting confidence bound from new model on risk ratio scale
        """

        if ci < 1:
            ci = 1 / ci
            new_ci = 1 / new_ci
        if ci < new_ci:
            ratio = new_ci / ci
        else:
            ratio = ci / new_ci

        return ratio + np.sqrt(ratio * (ratio - 1))

    def _evalue_OLS(self, est, se, sd, no_effect_baseline=0):
        """
        Computes E-value from OLS coefficient.

        :param est: coefficient from OLS
        :param se: standard error of point estimate
        :param sd: residual standard deviation
        :param no_effect_baseline: no_effect_baseline standardized difference to which to shift the observed estimate. (Default = 0)
        """
        if se < 0:
            raise ValueError("Standard error cannot be negative")

        delta = abs(self.estimate.estimator._treatment_value - self.estimate.estimator._control_value)
        est = self._to_smd(est, sd, delta=delta)
        se = self._to_smd(
            se, sd, delta=1
        )  # already multiplied by delta in LinearRegressionEstimator._estimate_std_error()
        return self._evalue_MD(est=est, se=se, no_effect_baseline=no_effect_baseline)

    def _evalue_MD(self, est, se=None, no_effect_baseline=0):
        """
        Computes E-value for standardized difference and its confidence limits.

        :param est: point estimate as standardized difference (i.e., Cohen's d)
        :param se: standard error of the point estimate
        :param no_effect_baseline: no_effect_baseline standardized difference to which to shift the observed estimate. (Default = 0)
        """
        if se < 0:
            raise ValueError("Standard error cannot be negative")

        if se is None:
            lo = None
            hi = None
        else:
            # see Table 2 and p.37 in https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf
            lo = np.exp(0.91 * est - 1.78 * se)
            hi = np.exp(0.91 * est + 1.78 * se)

        est = self._md_to_rr(est)
        no_effect_baseline = self._md_to_rr(no_effect_baseline)

        return self._evalue_RR(est=est, lo=lo, hi=hi, no_effect_baseline=no_effect_baseline)

    def _evalue_OR(self, est, lo=None, hi=None, rare=None, no_effect_baseline=1):
        """
        Computes E-value for an odds ratio and its confidence limits.

        :param est: point estimate
        :param lo: lower limit of confidence interval
        :param hi: upper limit of confidence interval
        :param rare: if outcome is rare (<15%)
        :param no_effect_baseline: the no_effect_baseline OR to which to shift the observed estimate. (Default = 1)
        """
        if est < 0:
            raise ValueError("Odds Ratio cannot be negative")

        est = self._or_to_rr(est, rare)
        if lo is not None:
            lo = self._or_to_rr(lo, rare)
        if hi is not None:
            hi = self._or_to_rr(hi, rare)
        no_effect_baseline = self._or_to_rr(no_effect_baseline, rare)

        return self._evalue_RR(est=est, lo=lo, hi=hi, no_effect_baseline=no_effect_baseline)

    def _evalue_RR(self, est, lo=None, hi=None, no_effect_baseline=1):
        """
        Computes E-value for a risk ratio or rate ratio and its confidence limits.

        :param est: point estimate
        :param lo: lower limit of confidence interval
        :param hi: upper limit of confidence interval
        :param no_effect_baseline: the no_effect_baseline RR to which to shift the observed estimate. (Default = 1)
        """
        if est < 0:
            raise ValueError("Risk/Rate Ratio cannot be negative")
        if no_effect_baseline < 0:
            raise ValueError("no_effect_baseline value is impossible")
        if no_effect_baseline != 1:
            self.logger.info(
                'You are calculating a "non-null" E-value, i.e., an E-value for the minimum amount of unmeasured '
                "confounding needed to move the estimate and confidence interval to your specified no_effect_baseline value "
                "rather than to the null value."
            )
        if lo is not None and hi is not None:
            if lo > hi:
                raise ValueError("Lower confidence limit should be less than upper confidence limit")
        if lo is not None and est < lo:
            raise ValueError("Point estimate should be inside confidence interval")
        if hi is not None and est > hi:
            raise ValueError("Point estimate should be inside confidence interval")

        e_est = self._threshold(est, no_effect_baseline=no_effect_baseline)
        e_lo = self._threshold(lo, no_effect_baseline=no_effect_baseline)
        e_hi = self._threshold(hi, no_effect_baseline=no_effect_baseline)

        # if CI crosses null, set its E-value to 1
        null_CI = False
        if est > no_effect_baseline and lo is not None:
            null_CI = lo < no_effect_baseline
        if est < no_effect_baseline and hi is not None:
            null_CI = hi > no_effect_baseline
        if null_CI:
            e_lo = np.float64(1)
            e_hi = np.float64(1)

        # only report E-value for CI limit closer to null
        if lo is not None or hi is not None:
            if est > no_effect_baseline:
                e_hi = None
            else:
                e_lo = None

        return {
            "converted_estimate": est,
            "converted_lower_ci": lo,
            "converted_upper_ci": hi,
            "evalue_estimate": e_est,
            "evalue_lower_ci": e_lo,
            "evalue_upper_ci": e_hi,
        }

    def _threshold(self, x, no_effect_baseline=1):
        """
        Computes E-value for single value of risk ratio.

        :param x: risk ratio
        :param no_effect_baseline: the no_effect_baseline RR to which to shift the observed estimate
        """
        if x is None:
            return None

        if x <= 1:
            x = 1 / x
            no_effect_baseline = 1 / no_effect_baseline

        if no_effect_baseline <= x:
            return (x + np.sqrt(x * (x - no_effect_baseline))) / no_effect_baseline
        else:
            ratio = no_effect_baseline / x
            return ratio + np.sqrt(ratio * (ratio - 1))

    def _to_smd(self, est, sd, delta=1):
        """
        Converts estimate to standardized mean difference.

        :param est: estimate
        :param sd: residual standard deviation or standard deviation of outcome
        :param delta: contrast of interest in the treatment/exposure
        """
        return est * delta / sd

    def _md_to_rr(self, est):
        """
        Converts standardized mean difference to approximate risk ratio.

        :param est: estimate
        """
        # see Table 2 and p.37 in https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf
        return np.exp(0.91 * est)

    def _hr_to_rr(self, est, rare):
        """
        Converts hazard ratio to approximate risk ratio.

        :param est: estimate
        :param rare: if outcome is rare (<15%)
        """
        if rare is None:
            raise ValueError('Must specify whether the rare outcome assumption can be made. Use argument "rare" =')
        if rare:
            return est
        else:
            return (1 - 0.5 ** np.sqrt(est)) / ((1 - 0.5 ** np.sqrt(1 / est)))

    def _or_to_rr(self, est, rare):
        """
        Converts odds ratio to approximate risk ratio.

        :param est: estimate
        :param rare: if outcome is rare (<15%)
        """
        if rare:
            return est
        else:
            return np.sqrt(est)

    def __str__(self):
        s = "Sensitivity Analysis to Unobserved Confounding using the E-value\n\n"
        s += f"Unadjusted Estimates of Treatment: {self.treatment_name}\n"
        s += f"Estimate (converted to risk ratio scale): {self.stats['converted_estimate']}\n"
        s += f"Lower 95% CI (converted to risk ratio scale): {self.stats['converted_lower_ci']}\n"
        s += f"Upper 95% CI (converted to risk ratio scale): {self.stats['converted_upper_ci']}\n\n"
        s += "Sensitivity Statistics: \n"
        s += f"E-value for point estimate: {self.stats['evalue_estimate']}\n"
        s += f"E-value for lower 95% CI: {self.stats['evalue_lower_ci']}\n"
        s += f"E-value for upper 95% CI: {self.stats['evalue_upper_ci']}\n"
        largest_obs_evalue = self.benchmarking_results["observed_covariate_e_value"].iloc[0]
        largest_obs_cov = self.benchmarking_results.index[0]
        s += f"Largest Observed Covariate E-value: {largest_obs_evalue} ({largest_obs_cov})\n\n"
        s += "Interpretation of results:\n"
        if self.stats["evalue_lower_ci"] is None:
            ci = self.stats["evalue_upper_ci"]
            direction = "decrease"
        else:
            ci = self.stats["evalue_lower_ci"]
            direction = "increase"
        s += (
            f"Unmeasured confounder(s) would have to be associated with a {self.stats['evalue_estimate'].round(2)}-fold {direction} "
            f"in the risk of {self.outcome_name}, and must be {self.stats['evalue_estimate'].round(2)} times more prevalent in "
            f"{self.treatment_name}, to explain away the observed point estimate.\n"
        )
        s += (
            f"Unmeasured confounder(s) would have to be associated with a {ci.round(2)}-fold {direction} "
            f"in the risk of {self.outcome_name}, and must be {ci.round(2)} times more prevalent in "
            f"{self.treatment_name}, to explain away the observed confidence interval."
        )
        return s
