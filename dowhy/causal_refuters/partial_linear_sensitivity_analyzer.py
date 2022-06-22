import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dowhy.utils.util import get_numeric_features
from dowhy.causal_refuters.reisz import ReiszRegressor, ReiszRepresenter, generate_moment_function, get_alpha_estimator, get_generic_regressor, create_polynomial_function


class PartialLinearSensitivityAnalyzer:
    """
    Class to perform sensitivity analysis for partially linear model
        :param estimator: estimator of the causal model
        :param num_splits: number of splits for cross validation. (default = 5)
        :param shuffle_data : shuffle data or not before splitting into folds (default = False)
        :param shuffle_random_seed: seed for randomly shuffling data
        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders
        :param benchmark_common_causes: names of variables for bounding strength of confounders
        :param significance_level: confidence interval for statistical inference(default = 0.05)
        :param frac_strength_treatment: strength of association between unobserved confounder and treatment compared to benchmark covariate
        :param frac_strength_outcome: strength of association between unobserved confounder and outcome compared to benchmark covariate
        :param alpha_s_param_dict: dictionary with parameters for finding alpha_s 
        :param g_s_estimator_list: list of estimator objects for finding g_s. These objects should have fit() and predict() functions.
        :param g_s_estimator_param_list: list of dictionaries with parameters for tuning respective estimators in "g_s_estimator_list". 
                                         The order of the dictionaries in the list should be consistent with the estimator objects order in "g_s_estimator_list"
        :param observed_common_causes: common causes dataframe
        :param outcome: outcome dataframe
        :param treatment: treatment dataframe
    """

    def __init__(self, estimator=None, num_splits=5, shuffle_data=False,
                 shuffle_random_seed=None,  reisz_polynomial_max_degree=3,
                 r2yu_tw=0.04, r2tu_w=0.03, significance_level=0.05, benchmark_common_causes=None,
                 frac_strength_treatment=None, frac_strength_outcome=None,
                 observed_common_causes=None, treatment=None, outcome=None, g_s_estimator_list=None, alpha_s_param_dict=None, g_s_estimator_param_list=None , **kwargs):
        self.estimator = estimator
        self.num_splits = num_splits
        self.shuffle_data = shuffle_data
        self.shuffle_random_seed = shuffle_random_seed
        self.reisz_polynomial_max_degree = reisz_polynomial_max_degree
        self.g_s_estimator_list = g_s_estimator_list
        self.g_s_estimator_param_list = g_s_estimator_param_list
        self.alpha_s_param_dict = alpha_s_param_dict
        self.r2yu_tw = r2yu_tw
        self.r2tu_w = r2tu_w
        self.significance_level = significance_level
        self.observed_common_causes = observed_common_causes
        self.treatment = treatment
        self.outcome = outcome
        self.benchmark_common_causes = benchmark_common_causes
        self.frac_strength_outcome = frac_strength_outcome
        self.frac_strength_treatment = frac_strength_treatment

        self.RV = None
        self.RV_alpha = None
        self.point_estimate = None
        self.standard_error = None
        self.theta_s = None
        self.nu_2 = None
        self.sigma_2 = None
        self.S2 = None
        self.neyman_orthogonal_score_outcome = None
        self.neyman_orthogonal_score_treatment = None
        self.neyman_orthogonal_score_theta = None

        self.r2t_w = 0  # Partial R^2 of treatment with observed common causes
        self.r2y_tw = 0  # Partial R^2 of outcome with treatment and observed common causes
        self.results = None

    def get_phi_lower_upper(self, Cg, Calpha):
        """
        Calculate lower and upper influence function (phi)

        :param Cg: measure of strength of confounding that omitted variables generate in outcome regression
        :param Calpha: measure of strength of confounding that omitted variables generate in treatment regression

        :returns : lower bound of phi, upper bound of phi
        """
        phi_lower = self.neyman_orthogonal_score_theta - ((Cg * Calpha) / (2 * self.S)) * (-(self.sigma_2 / (
            self.nu_2 ** 2)) * self.neyman_orthogonal_score_treatment + (1 / self.nu_2) * self.neyman_orthogonal_score_outcome)
        phi_upper = self.neyman_orthogonal_score_theta + ((Cg * Calpha) / (2 * self.S)) * (-(self.sigma_2 / (
            self.nu_2 ** 2)) * self.neyman_orthogonal_score_treatment + (1 / self.nu_2) * self.neyman_orthogonal_score_outcome)

        return phi_lower, phi_upper

    def get_confidence_levels(self, r2yu_tw, r2tu_w, significance_level):
        """
        Returns lower and upper bounds for different explanatory powers of unobserved confounders

        Y_residual  = Y - E[Y | X, T] (residualized outcome)
        T_residual  = T - E[T | X] (residualized treatment)
        theta = E[(Y - E[Y | X, T)(T - E[T | X] )] / E[(T - E[T | X]) ^ 2]
        σ² = E[(Y - E[Y | X, T]) ^ 2] (expected value of residual outcome)
        ν^2 = E[(T - E[T | X])^2] (expected value of residual treatment)
        ψ_θ = m(Ws , g) + (Y - g(Ws))α(Ws) - θ 
        ψ_σ² = (Y - g(Ws)) ^ 2 - σ²
        ψ_ν2 = (2m(Ws, α ) - α^2) - ν^2

        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders
        :param significance_level: confidence interval for statistical inference(default = 0.05)

        :returns lower_confidence_bound: lower limit of confidence bound of the estimate
        :returns upper_confidence_bound: upper limit of confidence bound of the estimate
        :returns bias: omitted variable bias for the confounding scenario
        """

        Cg2 = r2yu_tw  # Strength of confounding that omitted variables generate in outcome regression
        Cg = np.sqrt(Cg2)
        # Strength of confounding that omitted variables generate in treatment regression
        Calpha2 = r2tu_w / (1 - r2tu_w)
        Calpha = np.sqrt(Calpha2)
        self.S = np.sqrt(self.S2)

        bound = self.S2 * Cg2 * Calpha2
        bias = np.sqrt(bound)

        phi_lower, phi_upper = self.get_phi_lower_upper(Cg = Cg, Calpha = Calpha)

        expected_phi_lower = np.mean(phi_lower * phi_lower)
        expected_phi_upper = np.mean(phi_upper * phi_upper)

        n1 = phi_lower.shape[0]
        n2 = phi_upper.shape[0]

        stddev_lower = np.sqrt(expected_phi_lower / n1)
        stddev_upper = np.sqrt(expected_phi_upper / n2)

        theta_lower = self.theta_s - bias
        theta_upper = self.theta_s + bias

        if significance_level is not None:
            probability = scipy.stats.norm.ppf(1 - significance_level)
            lower_confidence_bound = theta_lower - probability * \
                np.sqrt(np.mean(stddev_lower * stddev_lower) +
                        np.var(theta_lower))
            upper_confidence_bound = theta_upper + probability * \
                np.sqrt(np.mean(stddev_upper * stddev_upper) +
                        np.var(theta_upper))

        else:
            lower_confidence_bound = theta_lower
            upper_confidence_bound = theta_upper

        return lower_confidence_bound, upper_confidence_bound, bias

    def calculate_robustness_value(self, alpha):
        """
                Function to compute the robustness value of estimate against the confounders
                :param alpha: confidence interval for statistical inference

                :returns: robustness value
        """
        for t_val in np.arange(0, 1, 0.01):
            lower_confidence_bound, _, _ = self.get_confidence_levels(
                r2yu_tw=t_val, r2tu_w=t_val, significance_level=alpha)
            if lower_confidence_bound <= 0:
                return t_val
        return t_val

    def perform_benchmarking(self, r2yu_tw, r2tu_w):
        """
        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders

        :returns: python dictionary storing values of r2tu_w, r2yu_tw, short estimate, bias, lower_ate_bound,upper_ate_bound, lower_confidence_bound, upper_confidence_bound
        """

        lower_confidence_bound, upper_confidence_bound, bias = self.get_confidence_levels(
            r2yu_tw=r2yu_tw, r2tu_w=r2tu_w, significance_level=0.05)
        lower_ate_bound, upper_ate_bound, bias = self.get_confidence_levels(
            r2yu_tw=r2yu_tw, r2tu_w=r2tu_w, significance_level=None)

        benchmarking_results = {
            'r2tu_w': r2tu_w,
            'r2yu_tw': r2yu_tw,
            'short estimate': self.theta_s,
            'bias': bias,
            'lower_ate_bound': lower_ate_bound,
            'upper_ate_bound': upper_ate_bound,
            'lower_confidence_bound': lower_confidence_bound,
            'upper_confidence_bound': upper_confidence_bound
        }

        return benchmarking_results

    def get_regression_partial_r2(self, X, Y, numeric_features, split_indices):
        """
        Calculates the pearson non parametric partial R^2 from a regression function.

        :param X: numpy array containing set of regressors 
        :param Y: outcome variable in regression
        :param numeric_features: list of indices of columns with numeric features
        :param split_indices: training and testing data indices obtained after cross folding

        :returns: partial R^2 value
        """
        regression_model = get_generic_regressor(cv=split_indices,
                                                X=X, Y=Y, max_degree=self.reisz_polynomial_max_degree,
                                                estimator_list=self.g_s_estimator_list,
                                                estimator_param_list=self.g_s_estimator_param_list,
                                                numeric_features=numeric_features
                                                ) 

        num_samples = X.shape[0]
        regression_pred = np.zeros(num_samples) 
        for train, test in split_indices:
            reg_fn_fit = regression_model.fit(X[train], Y[train])
            regression_pred[test] = reg_fn_fit.predict(X[test])

        partial_r2 = np.var(regression_pred) / np.var(Y)

        return partial_r2

    def compute_bounds(self, split_indices=None):
        """
        Computes the change in partial R^2 due to presence of unobserved confounders
        :param split_indices: training and testing data indices obtained after cross folding

        :returns delta_r2_y_wj: observed additive gains in explanatory power with outcome when including benchmark covariate  on regression equation
        :returns delta_r2t_wj: observed additive gains in explanatory power with treatment when including benchmark covariate  on regression equation
        """
        features = self.observed_common_causes.copy()
        treatment_df = self.treatment.copy()
        T = treatment_df.values.ravel()
        W = features.to_numpy()
        Y = self.outcome.copy()
        Y = Y.values.ravel()
        num_samples = W.shape[0]

        # common causes after removing the benchmark causes
        W_j_df = features.drop(self.benchmark_common_causes, axis=1)
        numeric_features = get_numeric_features(X=W_j_df)
        W_j = W_j_df.to_numpy()

        # Partial R^2 of treatment with observed common causes removing benchmark causes
        r2t_w_j = self.get_regression_partial_r2(X = W_j, Y = T, numeric_features = numeric_features, split_indices = split_indices)

        delta_r2t_wj = (self.r2t_w - r2t_w_j)

        # dataframe with treatment and observed common causes after removing benchmark causes
        T_W_j_df = pd.concat([treatment_df, W_j_df], axis=1)
        numeric_features = get_numeric_features(X=T_W_j_df)
        T_W_j = T_W_j_df.to_numpy()

        reg_function = get_generic_regressor(cv=split_indices,
                                             X=T_W_j, Y=Y, max_degree=self.reisz_polynomial_max_degree,
                                             estimator_list=self.g_s_estimator_list,
                                             estimator_param_list=self.g_s_estimator_param_list,
                                             numeric_features=numeric_features
                                             )  # Regressing over observed common causes removing benchmark causes and treatment

        reisz_function = get_alpha_estimator(cv=split_indices, X=T_W_j,
                                             max_degree=self.reisz_polynomial_max_degree, param_grid_dict=self.alpha_s_param_dict)

        for train, test in split_indices:
            reg_fn_fit = reg_function.fit(T_W_j[train], Y[train])
            self.g_s_j[test] = reg_fn_fit.predict(T_W_j[test])
            reisz_fn_fit = reisz_function.fit(T_W_j[train])
            self.alpha_s_j[test] = reisz_fn_fit.predict(T_W_j[test])

        # Partial R^2 of outcome with observed common causes and treatment after removing benchmark causes
        r2y_tw_j = np.var(self.g_s_j) / np.var(Y)
        delta_r2_y_wj = (self.r2y_tw - r2y_tw_j)

        return delta_r2_y_wj, delta_r2t_wj

    def check_sensitivity(self, plot=True):
        """
        Function to perform sensitivity analysis. 
        :param plot: plot = True generates a plot of lower confidence bound of the estimate for different variations of unobserved confounding.
                     plot = False overrides the setting

        :returns: instance of PartialLinearSensitivityAnalyzer class
        """

        self.point_estimate = self.estimator.intercept__inference().point_estimate
        self.standard_error = self.estimator.intercept__inference().stderr
        self.theta_s = self.point_estimate[0]

        features = self.observed_common_causes.copy()
        treatment_df = self.treatment.copy()
        X = pd.concat([treatment_df, features], axis=1)

        numeric_features = get_numeric_features(X)
        X = X.to_numpy()
        T = treatment_df.to_numpy()
        Y = self.outcome.copy()
        Y = Y.to_numpy()

        cv = KFold(n_splits=self.num_splits, shuffle=self.shuffle_data,
                   random_state=self.shuffle_random_seed)
        num_samples = X.shape[0]
        split_indices = list(cv.split(X))
        indices = np.arange(0, num_samples, 1)

        # tuple of residuals from first stage estimation, features and confounders
        residuals = self.estimator.residuals_

        residualized_outcome = residuals[0]
        residualized_treatment = residuals[1]

        n_residuals = residualized_outcome.shape[0]
        indices = np.arange(0, n_residuals, 1)

        residualized_outcome = residualized_outcome[indices]  # Y - g(Ws)
        residualized_treatment = residualized_treatment[indices]

        self.sigma_2 = np.mean(residualized_outcome * residualized_outcome)
        self.nu_2 = np.mean(residualized_treatment * residualized_treatment)

        self.S2 = self.sigma_2 / self.nu_2

        self.neyman_orthogonal_score_outcome = residualized_outcome * residualized_outcome - self.sigma_2
        self.neyman_orthogonal_score_treatment = residualized_treatment * residualized_treatment - self.nu_2
        self.neyman_orthogonal_score_theta = (residualized_outcome - residualized_treatment * self.theta_s) * residualized_treatment / self.nu_2

        self.g_s = Y - residualized_outcome
        self.alpha_s = (residualized_treatment) / np.mean(residualized_treatment * residualized_treatment)
        # Partial R^2 of treatment with observed common causes
        self.r2t_w = np.var(T - residualized_treatment) / np.var(T)
        self.r2y_tw = np.var(Y - residualized_outcome) / np.var(Y)

        self.g_s_j = np.zeros(num_samples)
        self.alpha_s_j = np.zeros(num_samples)

        delta_r2_y_wj, delta_r2t_wj = self.compute_bounds(
            split_indices=split_indices)

        # Partial R^2 of outcome after regressing over unobserved confounder, observed common causes and treatment
        r2y_uwt = self.frac_strength_outcome * delta_r2_y_wj + self.r2y_tw
        # Partial R^2 of treatment after regressing over unobserved confounder and observed common causes
        r2t_uw = self.frac_strength_treatment * delta_r2t_wj + self.r2t_w

        if r2y_uwt >=1:
            raise ValueError("r2y_uwt can not be >= 1. Try a lower effect_fraction_on_outcome value")
        if r2t_uw >= 1:
            raise ValueError("r2t_uw can not be >= 1. Try a lower effect_fraction_on_treatment value")

        self.r2yu_tw = abs((r2y_uwt - self.r2y_tw) / (1 - self.r2y_tw))
        self.r2tu_w = abs((r2t_uw - self.r2t_w) / (1 - self.r2t_w))

        if self.r2yu_tw >= 1:
            self.r2yu_tw = 1
            self.logger.warning("Warning: r2yu_tw can not be > 1. Try a lower effect_fraction_on_outcome. Setting r2yu_tw to 1")
        if self.r2tu_w >= 1:
            self.r2tu_w = 0.9999
            self.logger.warning("Warning: r2tu_w can not be > 1. Try a lower effect_fraction_on_treatment. Setting r2tu_w to 1")

        benchmarking_results = self.perform_benchmarking(
            r2yu_tw=self.r2yu_tw, r2tu_w=self.r2tu_w)
        self.results = pd.DataFrame(benchmarking_results, index=[0])

        self.RV = self.calculate_robustness_value(alpha=None)
        self.RV_alpha = self.calculate_robustness_value(
            alpha=self.significance_level)

        if plot == True:
            self.plot()

        return self

    def plot(self, plot_type="lower_confidence_bound", x_limit=0.8, y_limit=0.8,
             num_points_per_contour=30, plot_size=(7, 7), contours_color="blue", critical_contour_color="red",
             label_fontsize=9, contour_linewidths=0.75, contour_linestyles="solid",
             contours_label_color="black", critical_label_color="red",
             unadjusted_estimate_marker='D', unadjusted_estimate_color="black",
             adjusted_estimate_marker='^', adjusted_estimate_color="red",
             legend_position=(1.6, 0.6)):
        """
        Plots and summarizes the sensitivity bounds as a contour plot, as they vary with the partial R^2 of the unobserved confounder(s) with the treatment and the outcome
        Two types of plots can be generated, based on adjusted estimates or adjusted t-values
        X-axis: Partial R^2 of treatment and unobserved confounder(s)
        Y-axis: Partial R^2 of outcome and unobserved confounder(s)
        We also plot bounds on the partial R^2 of the unobserved confounders obtained from observed covariates

        :param plot_type: possible values are 'bias','lower_ate_bound','upper_ate_bound','lower_confidence_bound','upper_confidence_bound'
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
        critical_value = 0

        fig, ax = plt.subplots(1, 1, figsize=plot_size)
        ax.set_title("Sensitivity contour plot of %s" % plot_type)
        ax.set_xlabel("Partial R^2 of confounder with treatment")
        ax.set_ylabel("Partial R^2 of confounder with outcome")

        if(self.r2tu_w > 0.8 or self.r2yu_tw > 0.8):
            x_limit = 0.99
            y_limit = 0.99

        ax.set_xlim(-x_limit/20, x_limit)
        ax.set_ylim(-y_limit/20, y_limit)

        r2tu_w = np.arange(0.0, x_limit, x_limit / num_points_per_contour)
        r2yu_tw = np.arange(0.0, y_limit, y_limit / num_points_per_contour)

        undjusted_estimates = None
        contour_values = np.zeros((len(r2yu_tw), len(r2tu_w)))

        for i in range(len(r2yu_tw)):
            y = r2yu_tw[i]
            for j in range(len(r2tu_w)):
                x = r2tu_w[j]
                benchmarking_results = self.perform_benchmarking(
                    r2yu_tw=y, r2tu_w=x)
                contour_values[i][j] = benchmarking_results[plot_type]

        contour_plot = ax.contour(r2tu_w, r2yu_tw, contour_values, colors=contours_color,
                                  linewidths=contour_linewidths, linestyles=contour_linestyles)
        ax.clabel(contour_plot, inline=1, fontsize=label_fontsize,
                  colors=contours_label_color)

        if (critical_value >= contour_values.min() and critical_value <= contour_values.max()):
            contour_plot = ax.contour(r2tu_w, r2yu_tw, contour_values, colors=critical_contour_color,
                                      linewidths=contour_linewidths, levels=[critical_value])
            ax.clabel(contour_plot,  [critical_value], inline=1,
                      fontsize=label_fontsize, colors=critical_label_color)

        # Adding unadjusted point estimate
        if(plot_type == "lower_confidence_bound" or plot_type == "upper_confidence_bound" or plot_type == "lower_ate_bound" or plot_type == "upper_ate_bound"):
            ax.scatter([0], [0], marker=unadjusted_estimate_marker, color=unadjusted_estimate_color,
                       label="Unadjusted({:1.2f})".format(self.theta_s))

        # Adding bounds to partial R^2 values for given strength of confounders
        if(self.frac_strength_treatment == self.frac_strength_outcome):
            signs = str(round(self.frac_strength_treatment, 2))
        else:
            signs = str(round(self.frac_strength_treatment, 2)) + \
                '/' + str(round(self.frac_strength_outcome, 2))
        label = signs + ' X ' + \
            str(self.benchmark_common_causes) + \
            " ({:1.2f}) ".format(self.results[plot_type][0])
        ax.scatter(self.r2tu_w, self.r2yu_tw, color=adjusted_estimate_color,
                   marker=adjusted_estimate_marker, label=label)

        plt.margins()
        ax.legend(bbox_to_anchor=legend_position)
        plt.show()
