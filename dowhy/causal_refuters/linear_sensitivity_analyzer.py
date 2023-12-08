import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

from dowhy.utils.api import parse_state


class LinearSensitivityAnalyzer:
    """
    Class to perform sensitivity analysis
    See: https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20(2020)%20-%20Making%20Sense%20of%20Sensitivity.pdf

    :param estimator: linear estimator of the causal model
    :param data: Pandas dataframe
    :param treatment_name: name of treatment
     :param percent_change_estimate: It is the percentage of reduction of treatment estimate that could alter the results (default = 1)
                                        if percent_change_estimate = 1, the robustness value describes the strength of association of confounders with treatment and outcome in order to reduce the estimate by 100% i.e bring it down to 0.
    :param null_hypothesis_effect: assumed effect under the null hypothesis
    :param confounder_increases_estimate: True implies that confounder increases the absolute value of estimate and vice versa. (Default = True)
    :param benchmark_common_causes: names of variables for bounding strength of confounders
    :param significance_level: confidence interval for statistical inference(default = 0.05)
    :param frac_strength_treatment: strength of association between unobserved confounder and treatment compared to benchmark covariate
    :param frac_strength_outcome: strength of association between unobserved confounder and outcome compared to benchmark covariate
    :param common_causes_order: The order of column names in OLS regression data
    """

    def __init__(
        self,
        estimator=None,
        data=None,
        treatment_name=None,
        percent_change_estimate=1.0,
        significance_level=0.05,
        confounder_increases_estimate=True,
        benchmark_common_causes=None,
        null_hypothesis_effect=0,
        frac_strength_treatment=None,
        frac_strength_outcome=None,
        common_causes_order=None,
    ):
        self.data = data
        self.treatment_name = []
        # original_treatment_name: : stores original variable names for labelling
        self.original_treatment_name = treatment_name
        for t in range(len(treatment_name)):
            self.treatment_name.append("x" + str(t + 1))

        self.percent_change_estimate = percent_change_estimate
        self.significance_level = significance_level
        self.confounder_increases_estimate = confounder_increases_estimate
        self.estimator = estimator
        self.estimator_model = estimator.model
        self.null_hypothesis_effect = null_hypothesis_effect

        # common_causes_map : maps the original variable names to variable names in OLS regression
        self.common_causes_map = {}
        for i in range(len(common_causes_order)):
            self.common_causes_map[common_causes_order[i]] = "x" + str(len(self.treatment_name) + i + 1)

        # benchmark_common_causes: stores variable names in terms of regression model variables
        benchmark_common_causes = parse_state(benchmark_common_causes)
        self.benchmark_common_causes = []
        # original_benchmark_covariates: stores original variable names for labelling
        self.original_benchmark_covariates = benchmark_common_causes
        for i in range(len(benchmark_common_causes)):
            self.benchmark_common_causes.append(self.common_causes_map[benchmark_common_causes[i]])

        if type(frac_strength_treatment) in [int, list, float]:
            self.frac_strength_treatment = np.array(frac_strength_treatment)
        if type(frac_strength_outcome) in [int, list, float]:
            self.frac_strength_outcome = np.array(frac_strength_outcome)

        # estimate: estimate of regression
        self.estimate = None
        # degree_of_freedom: degree of freedom of error in regression
        self.degree_of_freedom = None
        # standard_error: standard error in regression
        self.standard_error = None
        # t_stats: Treatment coefficient t-value - measures how many standard errors the estimate is away from zero.
        self.t_stats = None
        # partial_f2: value to determine if a regression model and a nested version of it have a statistically significant difference between them
        self.partial_f2 = None
        # r2tu_w: partial R^2  of unobserved confounder "u" with treatment "t", after conditioning on observed covariates "w"
        self.r2tu_w = None
        # r2yu_tw: partial R^2  of unobserved confounder "u" with outcome "y", after conditioning on observed covariates "w" and treatment "t"
        self.r2yu_tw = None
        # r2twj_w: partial R^2 of observed covariate wj with treatment "t", after conditioning on observed covariates "w" excluding wj
        self.r2twj_w = None
        # r2ywj_tw:  partial R^2 of observed covariate wj with outcome "y", after conditioning on observed covariates "w" (excluding wj) and treatment "t"
        self.r2ywj_tw = None
        # benchmarking_results: dataframe containing information about bounds and bias adjusted terms
        self.benchmarking_results = None
        # stats: dictionary containing information like robustness value, partial R^2, estimate, standard error , degree of freedom, partial f^2, t-statistic
        self.stats = None
        self.logger = logging.getLogger(__name__)

    def treatment_regression(self):
        """
        Function to perform regression with treatment as outcome

        :returns: new OLS regression model
        """

        features = self.estimator._observed_common_causes.copy()
        treatment_df = self.data[self.original_treatment_name].copy()
        features = sm.tools.add_constant(features)
        features.rename(columns=self.common_causes_map, inplace=True)
        model = sm.OLS(treatment_df, features)
        estimator_model = model.fit()

        return estimator_model

    def partial_r2_func(self, estimator_model=None, treatment=None):
        """
        Computes the partial R^2 of regression model

        :param estimator_model: Linear regression model
        :param treatment: treatment name

        :returns: partial R^2 value
        """

        estimate = estimator_model.params[treatment]
        degree_of_freedom = int(estimator_model.df_resid)

        if np.isscalar(estimate):  # for single covariate
            t_stats = estimator_model.tvalues[treatment]
            return t_stats**2 / (t_stats**2 + degree_of_freedom)

        else:  # compute for a group of covariates
            covariance_matrix = estimator_model.cov_params().loc[treatment, :][treatment]
            n = len(estimate)  # number of parameters in model
            f_stat = (
                np.matmul(np.matmul(estimate.values.T, np.linalg.inv(covariance_matrix.values)), estimate.values) / n
            )
            return f_stat * n / (f_stat * n + degree_of_freedom)

    def robustness_value_func(self, alpha=1.0):
        """
        Function to calculate the robustness value.
        It is the minimum strength of association that confounders must have with treatment and outcome to change conclusions.
        Robustness value describes how strong the association must be in order to reduce the estimated effect by (100 * percent_change_estimate)%.
        Robustness value close to 1 means the treatment effect can handle strong confounders explaining  almost all residual variation of the treatment and the outcome.
        Robustness value close to 0 means that even very weak confounders can also change the results.

        :param alpha: confidence interval (default = 1)

        :returns: robustness value
        """

        partial_cohen_f = abs(
            self.t_stats / np.sqrt(self.degree_of_freedom)
        )  # partial f of treatment t with outcome y. f = t_val/sqrt(dof)
        f_q = self.percent_change_estimate * partial_cohen_f
        t_alpha_df_1 = t.ppf(
            alpha / 2, self.degree_of_freedom - 1
        )  # t-value threshold with alpha significance level and dof-1 degrees of freedom
        f_critical = abs(t_alpha_df_1) / np.sqrt(self.degree_of_freedom - 1)
        f_adjusted = f_q - f_critical

        if f_adjusted < 0:
            r_value = 0
        else:
            r_value = 0.5 * (np.sqrt(f_adjusted**4 + (4 * f_adjusted**2)) - f_adjusted**2)

        if f_adjusted > 0 and f_q > 1 / f_critical:
            r_value = (f_q**2 - f_critical**2) / (1 + f_q**2)

        return r_value

    def compute_bias_adjusted(self, r2tu_w, r2yu_tw):
        """
        Computes the bias adjusted estimate, standard error, t-value,  partial R2, confidence intervals

        :param r2tu_w: partial r^2 from regressing unobserved confounder u on treatment t after conditioning on observed covariates w
        :param r2yu_tw: partial r^2 from regressing unobserved confounder u on outcome y after conditioning on observed covariates w and treatment t

        :returns: Python dictionary with information about partial R^2 of confounders with treatment and outcome and bias adjusted variables
        """

        bias_factor = np.sqrt((r2yu_tw * r2tu_w) / (1 - r2tu_w))
        bias = bias_factor * (self.standard_error * np.sqrt(self.degree_of_freedom))

        if self.confounder_increases_estimate:
            bias_adjusted_estimate = np.sign(self.estimate) * (abs(self.estimate) - bias)
        else:
            bias_adjusted_estimate = np.sign(self.estimate) * (abs(self.estimate) + bias)

        bias_adjusted_se = (
            np.sqrt((1 - r2yu_tw) / (1 - r2tu_w))
            * self.standard_error
            * np.sqrt(self.degree_of_freedom / (self.degree_of_freedom - 1))
        )

        bias_adjusted_t = (bias_adjusted_estimate - self.null_hypothesis_effect) / bias_adjusted_se

        bias_adjusted_partial_r2 = bias_adjusted_t**2 / (
            bias_adjusted_t**2 + (self.degree_of_freedom - 1)
        )  # partial r2 formula used with new t value and dof - 1

        num_se = t.ppf(
            self.significance_level / 2, self.degree_of_freedom
        )  # Number of standard errors within Confidence Interval

        bias_adjusted_upper_CI = bias_adjusted_estimate - num_se * bias_adjusted_se
        bias_adjusted_lower_CI = bias_adjusted_estimate + num_se * bias_adjusted_se

        benchmarking_results = {
            "r2tu_w": r2tu_w,
            "r2yu_tw": r2yu_tw,
            "bias_adjusted_estimate": bias_adjusted_estimate,
            "bias_adjusted_se": bias_adjusted_se,
            "bias_adjusted_t": bias_adjusted_t,
            "bias_adjusted_lower_CI": bias_adjusted_lower_CI,
            "bias_adjusted_upper_CI": bias_adjusted_upper_CI,
        }

        return benchmarking_results

    def check_sensitivity(self, plot=True):
        """
        Function to perform sensitivity analysis.
        :param plot: plot = True generates a plot of point estimate and the variations with respect to unobserved confounding.
                     plot = False overrides the setting

        :returns: instance of LinearSensitivityAnalyzer class
        """

        self.standard_error = np.array(self.estimator_model.bse[1 : (len(self.treatment_name) + 1)])[0]
        self.degree_of_freedom = int(self.estimator_model.df_resid)
        self.estimate = np.array(self.estimator_model.params[1 : (len(self.treatment_name) + 1)])[0]
        self.t_stats = np.array(self.estimator_model.tvalues[self.treatment_name])[0]

        # partial R^2 (r2yt_w) is the proportion of variation in outcome uniquely explained by treatment
        partial_r2 = self.partial_r2_func(self.estimator_model, self.treatment_name)
        RVq = self.robustness_value_func()
        RV_qalpha = self.robustness_value_func(alpha=self.significance_level)

        if self.confounder_increases_estimate:
            self.null_hypothesis_effect = self.estimate * (1 - self.percent_change_estimate)
        else:
            self.null_hypothesis_effect = self.estimate * (1 + self.percent_change_estimate)

        self.t_stats = (self.estimate - self.null_hypothesis_effect) / self.standard_error
        self.partial_f2 = self.t_stats**2 / self.degree_of_freedom

        # build a new regression model by considering treatment variables as outcome
        treatment_linear_model = self.treatment_regression()

        # r2twj_w is partial R^2 of covariate wj with treatment "t", after conditioning on covariates w(excluding wj)
        # r2ywj_tw is partial R^2 of covariate wj with outcome "y", after conditioning on covariates w(excluding wj) and treatment "t"
        self.r2twj_w = []
        self.r2ywj_tw = []

        for covariate in self.benchmark_common_causes:
            self.r2ywj_tw.append(self.partial_r2_func(self.estimator_model, covariate))
            self.r2twj_w.append(self.partial_r2_func(treatment_linear_model, covariate))

        for i in range(len(self.benchmark_common_causes)):
            r2twj_w = self.r2twj_w[i]
            r2ywj_tw = self.r2ywj_tw[i]

            # r2tu_w is the partial r^2 from regressing u on t after conditioning on w
            self.r2tu_w = self.frac_strength_treatment * (r2twj_w / (1 - r2twj_w))
            if any(val >= 1 for val in self.r2tu_w):
                raise ValueError("r2tu_w can not be >= 1. Try a lower frac_strength_treatment value")

            r2uwj_wt = (
                self.frac_strength_treatment
                * (r2twj_w**2)
                / ((1 - self.frac_strength_treatment * r2twj_w) * (1 - r2twj_w))
            )
            if any(val >= 1 for val in r2uwj_wt):
                raise ValueError("r2uwj_wt can not be >= 1. Try a lower frac_strength_treatment value")

            self.r2yu_tw = ((np.sqrt(self.frac_strength_outcome) + np.sqrt(r2uwj_wt)) / np.sqrt(1 - r2uwj_wt)) ** 2 * (
                r2ywj_tw / (1 - r2ywj_tw)
            )
            if any(val > 1 for val in self.r2yu_tw):
                for i in range(len(self.r2yu_tw)):
                    if self.r2yu_tw[i] > 1:
                        self.r2yu_tw[i] = 1
                self.logger.warning(
                    "Warning: r2yu_tw can not be > 1. Try a lower frac_strength_treatment. Setting r2yu_tw to 1"
                )

            # Compute bias adjusted terms

        self.benchmarking_results = self.compute_bias_adjusted(self.r2tu_w, self.r2yu_tw)

        if plot == True:
            self.plot()

        self.stats = {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "degree of freedom": self.degree_of_freedom,
            "t_statistic": self.t_stats,
            "r2yt_w": partial_r2,
            "partial_f2": self.partial_f2,
            "robustness_value": RVq,
            "robustness_value_alpha": RV_qalpha,
        }

        self.benchmarking_results = pd.DataFrame.from_dict(self.benchmarking_results)
        return self

    def plot_estimate(self, r2tu_w, r2yu_tw):
        """
        Computes the contours, threshold line and bounds for plotting estimates.
        Contour lines (z - axis) correspond to the adjusted estimate values for different values of r2tu_w (x) and r2yu_tw (y).
        :param r2tu_w: hypothetical partial R^2 of confounder with treatment(x - axis)
        :param r2yu_tw: hypothetical partial R^2 of confounder with outcome(y - axis)

        :returns:
        contour_values : values of contour lines for the plot
        critical_estimate : threshold point
        estimate_bounds : estimate values for unobserved confounders (bias adjusted estimates)
        """

        critical_estimate = self.null_hypothesis_effect
        contour_values = np.zeros((len(r2yu_tw), len(r2tu_w)))
        for i in range(len(r2yu_tw)):
            y = r2yu_tw[i]
            for j in range(len(r2tu_w)):
                x = r2tu_w[j]
                benchmarking_results = self.compute_bias_adjusted(r2tu_w=x, r2yu_tw=y)
                estimate = benchmarking_results["bias_adjusted_estimate"]
                contour_values[i][j] = estimate

        estimate_bounds = self.benchmarking_results["bias_adjusted_estimate"]
        return contour_values, critical_estimate, estimate_bounds

    def plot_t(self, r2tu_w, r2yu_tw):
        """
        Computes the contours, threshold line and bounds for plotting t.
        Contour lines (z - axis) correspond to the adjusted t values for different values of r2tu_w (x) and r2yu_tw (y).
        :param r2tu_w: hypothetical partial R^2 of confounder with treatment(x - axis)
        :param r2yu_tw: hypothetical partial R^2 of confounder with outcome(y - axis)

        :returns:
        contour_values : values of contour lines for the plot
        critical_t : threshold point
        t_bounds : t-value for unobserved confounders (bias adjusted t values)
        """

        t_alpha_df_1 = t.ppf(
            self.significance_level / 2, self.degree_of_freedom - 1
        )  # t-value threshold with alpha significance level and dof-1 degrees of freedom
        critical_t = abs(t_alpha_df_1) * np.sign(self.t_stats)

        contour_values = []
        for x in r2tu_w:
            contour = []
            for y in r2yu_tw:
                benchmarking_results = self.compute_bias_adjusted(r2tu_w=x, r2yu_tw=y)
                t_value = benchmarking_results["bias_adjusted_t"]
                contour.append(t_value)
            contour_values.append(contour)

        t_bounds = self.benchmarking_results["bias_adjusted_t"]
        return contour_values, critical_t, t_bounds

    def plot(
        self,
        plot_type="estimate",
        critical_value=None,
        x_limit=0.8,
        y_limit=0.8,
        num_points_per_contour=200,
        plot_size=(7, 7),
        contours_color="blue",
        critical_contour_color="red",
        label_fontsize=9,
        contour_linewidths=0.75,
        contour_linestyles="solid",
        contours_label_color="black",
        critical_label_color="red",
        unadjusted_estimate_marker="D",
        unadjusted_estimate_color="black",
        adjusted_estimate_marker="^",
        adjusted_estimate_color="red",
        legend_position=(1.6, 0.6),
    ):
        """
        Plots and summarizes the sensitivity bounds as a contour plot, as they vary with the partial R^2 of the unobserved confounder(s) with the treatment and the outcome
        Two types of plots can be generated, based on adjusted estimates or adjusted t-values
        X-axis: Partial R^2 of treatment and unobserved confounder(s)
        Y-axis: Partial R^2 of outcome and unobserved confounder(s)
        We also plot bounds on the partial R^2 of the unobserved confounders obtained from observed covariates

        :param plot_type: "estimate" or "t-value"
        :param critical_value: special reference value of the estimate or t-value that will be highlighted in the plot
        :param x_limit: plot's maximum x_axis value (default = 0.8)
        :param y_limit: plot's minimum y_axis value (default = 0.8)
        :param num_points_per_contour: number of points to calculate and plot each contour line (default = 200)
        :param plot_size: tuple denoting the size of the plot (default = (7,7))
        :param contours_color: color of contour line (default = blue)
                               String or array. If array, lines will be plotted with the specific color in ascending order.
        :param critical_contour_color: color of threshold line (default = red)
        :param label_fontsize: fontsize for labelling contours (default = 9)
        :param contour_linewidths: linewidths for contours (default = 0.75)
        :param contour_linestyles: linestyles for contours (default = "solid")
                                   See : https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/linestyles.html for more examples
        :param contours_label_color: color of contour line label (default = black)
        :param critical_label_color: color of threshold line label (default = red)
        :param unadjusted_estimate_marker: marker type for unadjusted estimate in the plot (default = 'D')
                                           See: https://matplotlib.org/stable/api/markers_api.html
        :parm unadjusted_estimate_color: marker color for unadjusted estimate in the plot (default = "black")
        :param adjusted_estimate_marker: marker type for bias adjusted estimates in the plot (default = '^')
        :parm adjusted_estimate_color: marker color for bias adjusted estimates in the plot (default = "red")
        :param legend_position:tuple denoting the position of the legend (default = (1.6, 0.6))
        """

        # Plotting the contour plot
        if plot_type == "estimate":
            critical_value = 0  # default value of estimate
        else:
            critical_value = 2  # default t-value (usual approx for 95% CI)

        fig, ax = plt.subplots(1, 1, figsize=plot_size)
        ax.set_title("Sensitivity contour plot of %s" % plot_type)
        ax.set_xlabel("Partial R^2 of confounder with treatment")
        ax.set_ylabel("Partial R^2 of confounder with outcome")

        for i in range(len(self.r2tu_w)):
            x = self.r2tu_w[i]
            y = self.r2yu_tw[i]
            if x > 0.8 or y > 0.8:
                x_limit = 0.99
                y_limit = 0.99
                break

        r2tu_w = np.arange(0.0, x_limit, x_limit / num_points_per_contour)
        r2yu_tw = np.arange(0.0, y_limit, y_limit / num_points_per_contour)

        unadjusted_point_estimate = None

        if plot_type == "estimate":
            contour_values, critical_value, bound_values = self.plot_estimate(r2tu_w, r2yu_tw)
            unadjusted_estimate = self.estimate
            unadjusted_point_estimate = unadjusted_estimate
        elif plot_type == "t-value":
            contour_values, critical_value, bound_values = self.plot_t(r2tu_w, r2yu_tw)
            unadjusted_t = self.t_stats
            unadjusted_point_estimate = unadjusted_t
        else:
            raise ValueError("Current plotting method only supports 'estimate' and 't-value' ")

        # Adding contours
        contour_plot = ax.contour(
            r2tu_w,
            r2yu_tw,
            contour_values,
            colors=contours_color,
            linewidths=contour_linewidths,
            linestyles=contour_linestyles,
        )
        ax.clabel(contour_plot, inline=1, fontsize=label_fontsize, colors=contours_label_color)

        # Adding threshold contour line
        contour_plot = ax.contour(
            r2tu_w,
            r2yu_tw,
            contour_values,
            colors=critical_contour_color,
            linewidths=contour_linewidths,
            levels=[critical_value],
        )
        ax.clabel(contour_plot, [critical_value], inline=1, fontsize=label_fontsize, colors=critical_label_color)

        # Adding unadjusted point estimate
        ax.scatter(
            [0],
            [0],
            marker=unadjusted_estimate_marker,
            color=unadjusted_estimate_color,
            label="Unadjusted({:1.2f})".format(unadjusted_point_estimate),
        )

        # Adding bounds to partial R^2 values for given strength of confounders
        for i in range(len(self.frac_strength_treatment)):
            frac_strength_treatment = self.frac_strength_treatment[i]
            frac_strength_outcome = self.frac_strength_outcome[i]
            if frac_strength_treatment == frac_strength_outcome:
                signs = str(round(frac_strength_treatment, 2))
            else:
                signs = str(round(frac_strength_treatment, 2)) + "/" + str(round(frac_strength_outcome, 2))
            label = (
                str(i + 1)
                + "  "
                + signs
                + " X "
                + str(self.original_benchmark_covariates)
                + " ({:1.2f}) ".format(bound_values[i])
            )
            ax.scatter(
                self.r2tu_w[i],
                self.r2yu_tw[i],
                color=adjusted_estimate_color,
                marker=adjusted_estimate_marker,
                label=label,
            )
            ax.annotate(str(i + 1), (self.r2tu_w[i] + 0.005, self.r2yu_tw[i] + 0.005))

        ax.legend(bbox_to_anchor=legend_position)
        plt.show()

    def __str__(self):
        s = "Sensitivity Analysis to Unobserved Confounding using R^2 paramterization\n\n"
        s += "Unadjusted Estimates of Treatment {0} :\n".format(self.original_treatment_name)
        s += "Coefficient Estimate : {0}\n".format(self.estimate)
        s += "Degree of Freedom : {0}\n".format(self.degree_of_freedom)
        s += "Standard Error : {0}\n".format(self.standard_error)
        s += "t-value : {0}\n".format(self.t_stats)
        s += "F^2 value : {0}\n\n".format(self.partial_f2)
        s += "Sensitivity Statistics : \n"
        s += "Partial R2 of treatment with outcome : {0}\n".format(self.stats["r2yt_w"])
        s += "Robustness Value : {0}\n\n".format(self.stats["robustness_value"])
        s += "Interpretation of results :\n"
        s += "Any confounder explaining less than {0}% percent of the residual variance of both the treatment and the outcome would not be strong enough to explain away the observed effect i.e bring down the estimate to 0 \n\n".format(
            round(self.stats["robustness_value"] * 100, 2)
        )
        s += "For a significance level of {0}%, any confounder explaining more than {1}% percent of the residual variance of both the treatment and the outcome would be strong enough to make the estimated effect not 'statistically significant'\n\n".format(
            self.significance_level * 100, round(self.stats["robustness_value_alpha"] * 100, 2)
        )
        s += "If confounders explained 100% of the residual variance of the outcome, they would need to explain at least {0}% of the residual variance of the treatment to bring down the estimated effect to 0\n".format(
            round(self.stats["r2yt_w"] * 100, 2)
        )
        return s
