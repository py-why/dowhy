import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dowhy.causal_refuters.reisz import get_generic_regressor
from dowhy.utils.regression import get_numeric_features


class PartialLinearSensitivityAnalyzer:
    """
    Class to perform sensitivity analysis for partially linear model.

    An efficient version of the non parametric sensitivity analyzer that works for estimators that return residuals of regression from confounders on treatment and outcome, such as the DML method. For all other methods (or when the partially linear assumption is not guaranteed to be satisfied), use the non-parametric sensitivity analysis.

    Based on this work:
        Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A., & Syrgkanis, V. (2022). Long Story Short: Omitted Variable Bias in Causal Machine Learning (No. w30302). National Bureau of Economic Research.

    :param estimator: estimator of the causal model
    :param num_splits: number of splits for cross validation. (default = 5)
    :param shuffle_data : shuffle data or not before splitting into folds (default = False)
    :param shuffle_random_seed: seed for randomly shuffling data
    :param effect_strength_treatment: C^2_T, list of plausible sensitivity parameters for effect of confounder on treatment
    :param effect_strength_outcome: C^2_Y, list of plausible sensitivity parameters for effect of confounder on outcome
    :param benchmark_common_causes: names of variables for bounding strength of confounders
    :param significance_level: confidence interval for statistical inference(default = 0.05)
    :param frac_strength_treatment: strength of association between unobserved confounder and treatment compared to benchmark covariate
    :param frac_strength_outcome: strength of association between unobserved confounder and outcome compared to benchmark covariate
    :param g_s_estimator_list: list of estimator objects for finding g_s. These objects should have fit() and predict() functions.
    :param g_s_estimator_param_list: list of dictionaries with parameters for tuning respective estimators in "g_s_estimator_list".
    :param alpha_s_estimator_list: list of estimator objects for finding the treatment predictor which is used for alpha_s estimation. These objects should have fit() and predict_proba() functions.
    :param alpha_s_estimator_param_list: list of dictionaries with parameters for tuning respective estimators in "alpha_s_estimator_list".
                                     The order of the dictionaries in the list should be consistent with the estimator objects order in "g_s_estimator_list"
    :param observed_common_causes: common causes dataframe
    :param outcome: outcome dataframe
    :param treatment: treatment dataframe
    """

    def __init__(
        self,
        estimator=None,
        num_splits=5,
        shuffle_data=False,
        shuffle_random_seed=None,
        reisz_polynomial_max_degree=3,
        significance_level=0.05,
        effect_strength_treatment=None,
        effect_strength_outcome=None,
        benchmark_common_causes=None,
        frac_strength_treatment=None,
        frac_strength_outcome=None,
        observed_common_causes=None,
        treatment=None,
        outcome=None,
        g_s_estimator_list=None,
        alpha_s_estimator_list=None,
        g_s_estimator_param_list=None,
        alpha_s_estimator_param_list=None,
        **kwargs,
    ):
        self.estimator = estimator
        self.num_splits = num_splits
        self.shuffle_data = shuffle_data
        self.shuffle_random_seed = shuffle_random_seed
        self.reisz_polynomial_max_degree = reisz_polynomial_max_degree
        self.effect_strength_treatment = effect_strength_treatment
        self.effect_strength_outcome = effect_strength_outcome
        self.g_s_estimator_list = g_s_estimator_list
        self.g_s_estimator_param_list = g_s_estimator_param_list
        self.alpha_s_estimator_list = alpha_s_estimator_list
        self.alpha_s_estimator_param_list = alpha_s_estimator_param_list
        self.significance_level = significance_level
        self.observed_common_causes = observed_common_causes
        self.treatment = treatment
        self.outcome = outcome
        self.benchmark_common_causes = benchmark_common_causes
        self.frac_strength_outcome = frac_strength_outcome
        self.frac_strength_treatment = frac_strength_treatment

        # whether the DGP is assumed to be partially linear
        self.is_partial_linear = True

        self.RV = None
        self.RV_alpha = None
        self.point_estimate = None
        self.standard_error = None
        self.theta_s = None
        self.nu_2 = None
        self.sigma_2 = None
        self.S2 = None
        self.S = None
        self.neyman_orthogonal_score_outcome = None
        self.neyman_orthogonal_score_treatment = None
        self.neyman_orthogonal_score_theta = None

        self.r2t_w = 0  # Partial R^2 of treatment with observed common causes
        self.r2y_tw = 0  # Partial R^2 of outcome with treatment and observed common causes
        self.results = None
        self.num_points_per_contour = 30

        self.benchmarking = self.is_benchmarking_needed()
        self.logger = logging.getLogger(__name__)

    def is_benchmarking_needed(self):
        # can change this to allow default values that are same as the other parameter
        if self.effect_strength_treatment is not None:
            if self.effect_strength_outcome is None:
                raise ValueError(
                    "Need to specify both partial_r2_confounder_treatment and partial_r2_confounder_outcome."
                )
        else:
            if self.effect_strength_outcome is not None:
                raise ValueError(
                    "Need to specify both partial_r2_confounder_treatment and partial_r2_confounder_outcome."
                )
        if self.benchmark_common_causes is not None:
            if self.frac_strength_outcome is not None or self.frac_strength_treatment is not None:
                return True
            else:
                raise ValueError(
                    "Need to specify at least one of effect_fraction_on_treatment or effect_fraction_on_outcome."
                )
        else:
            return False

    def get_phi_lower_upper(self, Cg, Calpha):
        """
        Calculate lower and upper influence function (phi)

        :param Cg: measure of strength of confounding that omitted variables generate in outcome regression
        :param Calpha: measure of strength of confounding that omitted variables generate in treatment regression

        :returns : lower bound of phi, upper bound of phi
        """
        phi_lower = self.neyman_orthogonal_score_theta - ((Cg * Calpha) / (2 * self.S)) * (
            -(self.sigma_2 / (self.nu_2**2)) * self.neyman_orthogonal_score_treatment
            + (1 / self.nu_2) * self.neyman_orthogonal_score_outcome
        )
        phi_upper = self.neyman_orthogonal_score_theta + ((Cg * Calpha) / (2 * self.S)) * (
            -(self.sigma_2 / (self.nu_2**2)) * self.neyman_orthogonal_score_treatment
            + (1 / self.nu_2) * self.neyman_orthogonal_score_outcome
        )

        return phi_lower, phi_upper

    def get_confidence_levels(self, r2yu_tw, r2tu_w, significance_level, is_partial_linear):
        """
        Returns lower and upper bounds for the effect estimate, given different explanatory powers of unobserved confounders. It uses the following definitions.

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
        :param is_partial_linear: whether the data-generating process is assumed to be partially linear

        :returns lower_confidence_bound: lower limit of confidence bound of the estimate
        :returns upper_confidence_bound: upper limit of confidence bound of the estimate
        :returns bias: omitted variable bias for the confounding scenario
        """

        Cg2 = r2yu_tw  # Strength of confounding that omitted variables generate in outcome regression

        # Strength of confounding that omitted variables generate in treatment regression
        Calpha2 = r2tu_w / (1 - r2tu_w)
        Cg = np.sqrt(Cg2)
        Calpha = np.sqrt(Calpha2)
        self.S = np.sqrt(self.S2)

        # computing the point estimate for the bounds
        bound = self.S2 * Cg2 * Calpha2
        bias = np.sqrt(bound)
        theta_lower = self.theta_s - bias
        theta_upper = self.theta_s + bias

        if significance_level is not None:
            phi_lower, phi_upper = self.get_phi_lower_upper(Cg=Cg, Calpha=Calpha)

            expected_phi_lower = np.mean(phi_lower * phi_lower)
            expected_phi_upper = np.mean(phi_upper * phi_upper)

            n1 = phi_lower.shape[0]
            n2 = phi_upper.shape[0]

            stddev_lower = np.sqrt(expected_phi_lower / n1)
            stddev_upper = np.sqrt(expected_phi_upper / n2)
            probability = scipy.stats.norm.ppf(1 - significance_level)
            lower_confidence_bound = theta_lower - probability * np.sqrt(
                np.mean(stddev_lower * stddev_lower) + np.var(theta_lower)
            )
            upper_confidence_bound = theta_upper + probability * np.sqrt(
                np.mean(stddev_upper * stddev_upper) + np.var(theta_upper)
            )

        else:
            lower_confidence_bound = theta_lower
            upper_confidence_bound = theta_upper

        return lower_confidence_bound, upper_confidence_bound, bias

    def calculate_robustness_value(self, alpha, is_partial_linear):
        """
        Function to compute the robustness value of estimate against the confounders
        :param alpha: confidence interval for statistical inference

        :returns: robustness value
        """
        for t_val in np.arange(0, 1, 0.01):
            lower_confidence_bound, _, _ = self.get_confidence_levels(
                r2yu_tw=t_val, r2tu_w=t_val, significance_level=alpha, is_partial_linear=is_partial_linear
            )
            if lower_confidence_bound <= 0:
                return t_val
        return t_val

    def perform_benchmarking(self, r2yu_tw, r2tu_w, significance_level, is_partial_linear=True):
        """
        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders
        :param significance_level: the desired significance level for the bounds
        :param is_partial_linear: whether we assume a partially linear data-generating process


        :returns: python dictionary storing values of r2tu_w, r2yu_tw, short estimate, bias, lower_ate_bound,upper_ate_bound, lower_confidence_bound, upper_confidence_bound
        """
        max_r2yu_tw = max(r2yu_tw) if np.ndim(r2yu_tw) != 0 else r2yu_tw
        max_r2tu_w = max(r2tu_w) if np.ndim(r2yu_tw) != 0 else r2tu_w
        lower_confidence_bound, upper_confidence_bound, bias = self.get_confidence_levels(
            r2yu_tw=max_r2yu_tw,
            r2tu_w=max_r2tu_w,
            significance_level=significance_level,
            is_partial_linear=is_partial_linear,
        )
        lower_ate_bound, upper_ate_bound, bias = self.get_confidence_levels(
            r2yu_tw=max_r2yu_tw, r2tu_w=max_r2tu_w, significance_level=None, is_partial_linear=is_partial_linear
        )

        benchmarking_results = {
            "r2tu_w": max_r2tu_w,
            "r2yu_tw": max_r2yu_tw,
            "short estimate": self.theta_s,
            "bias": bias,
            "lower_ate_bound": lower_ate_bound,
            "upper_ate_bound": upper_ate_bound,
            "lower_confidence_bound": lower_confidence_bound,
            "upper_confidence_bound": upper_confidence_bound,
        }

        return benchmarking_results

    def get_regression_r2(self, X, Y, numeric_features, split_indices, regression_model=None):
        """
        Calculates the pearson non parametric partial R^2 from a regression function.

        :param X: numpy array containing set of regressors
        :param Y: outcome variable in regression
        :param numeric_features: list of indices of columns with numeric features
        :param split_indices: training and testing data indices obtained after cross folding

        :returns: partial R^2 value
        """
        if regression_model is None:
            regression_model = get_generic_regressor(
                cv=split_indices,
                X=X,
                Y=Y,
                max_degree=self.reisz_polynomial_max_degree,
                estimator_list=self.g_s_estimator_list,
                estimator_param_list=self.g_s_estimator_param_list,
                numeric_features=numeric_features,
            )

        num_samples = X.shape[0]
        regression_pred = np.zeros(num_samples)
        for train, test in split_indices:
            reg_fn_fit = regression_model.fit(X[train], Y[train])
            regression_pred[test] = reg_fn_fit.predict(X[test])

        r2 = np.var(regression_pred) / np.var(Y)

        return r2

    def compute_r2diff_benchmarking_covariates(
        self,
        treatment_df,
        features,
        T,
        Y,
        W,
        benchmark_common_causes,
        split_indices=None,
        second_stage_linear=False,
        is_partial_linear=True,
    ):
        """
        Computes the change in partial R^2 due to presence of unobserved confounders
        :param split_indices: training and testing data indices obtained after cross folding
        :param second_stage_linear: True if second stage regression is linear else False (default = False)
        :param is_partial_linear: True if the data-generating process is assumed to be partially linear

        :returns delta_r2_y_wj: observed additive gains in explanatory power with outcome when including benchmark covariate  on regression equation
        :returns delta_r2t_wj: observed additive gains in explanatory power with treatment when including benchmark covariate  on regression equation
        """
        T = T.ravel()
        Y = Y.ravel()
        num_samples = W.shape[0]

        # common causes after removing the benchmark causes
        W_j_df = features.drop(benchmark_common_causes, axis=1)
        numeric_features = get_numeric_features(X=W_j_df)
        W_j = W_j_df.to_numpy()
        # dataframe with treatment and observed common causes after removing benchmark causes
        T_W_j_df = pd.concat([treatment_df, W_j_df], axis=1)
        numeric_features_t = get_numeric_features(X=T_W_j_df)
        T_W_j = T_W_j_df.to_numpy()

        # R^2 of treatment with observed common causes removing benchmark causes
        if is_partial_linear:
            r2t_w_j = self.get_regression_r2(X=W_j, Y=T, numeric_features=numeric_features, split_indices=split_indices)
            delta_r2t_wj = self.r2t_w - r2t_w_j
        else:  # non parametric DGP
            # return the variance of alpha_s
            var_alpha_wj = self.get_alpharegression_var(
                X=T_W_j,
                numeric_features=numeric_features,  # using numeric_features because the model only uses W
                split_indices=split_indices,
            )
            delta_r2t_wj = var_alpha_wj

        reg_function = None
        if second_stage_linear is True:
            reg_function = get_generic_regressor(
                cv=split_indices,
                X=T_W_j,
                Y=Y,
                max_degree=self.reisz_polynomial_max_degree,
                estimator_list=[
                    LinearRegression(),
                    Pipeline(
                        [
                            (
                                "scale",
                                ColumnTransformer(
                                    [("num", StandardScaler(), numeric_features)], remainder="passthrough"
                                ),
                            ),
                            ("lasso_model", Lasso()),
                        ]
                    ),
                    SGDRegressor(alpha=0.001),
                    Ridge(),
                    RidgeCV(cv=5),
                ],
                estimator_param_list=[
                    {"fit_intercept": [True, False]},
                    {"lasso_model__alpha": [0.01, 0.001, 1e-4, 1e-5, 1e-6]},
                    {"alpha": [0.0001, 1e-5, 0.01]},
                    {"alpha": [0.0001, 1e-5, 0.01, 1, 2]},
                    {"cv": [2, 3, 4]},
                ],
                numeric_features=numeric_features,
            )  # Regressing over observed common causes removing benchmark causes and treatment
        # R^2 of outcome with observed common causes and treatment after removing benchmark causes
        r2y_tw_j = self.get_regression_r2(
            X=T_W_j,
            Y=Y,
            numeric_features=numeric_features_t,
            split_indices=split_indices,
            regression_model=reg_function,
        )
        delta_r2_y_wj = self.r2y_tw - r2y_tw_j

        return delta_r2_y_wj, delta_r2t_wj

    def check_sensitivity(self, plot=True):
        """
        Function to perform sensitivity analysis.

        :param plot: plot = True generates a plot of lower confidence bound of the estimate for different variations of unobserved confounding.
                     plot = False overrides the setting

        :returns: instance of PartialLinearSensitivityAnalyzer class
        """

        # Obtaining theta_s (the obtained estimate)
        self.point_estimate = self.estimator.intercept__inference().point_estimate
        self.standard_error = self.estimator.intercept__inference().stderr
        self.theta_s = self.point_estimate[0]

        # Creating numpy arrays
        features = self.observed_common_causes.copy()
        treatment_df = self.treatment.copy()
        X_df = pd.concat([treatment_df, features], axis=1)
        W = features.to_numpy()
        numeric_features = get_numeric_features(X_df)
        X = X_df.to_numpy()
        T = treatment_df.to_numpy()
        Y = self.outcome.copy()
        Y = Y.to_numpy()

        # Setting up cross-validation parameters
        cv = KFold(n_splits=self.num_splits, shuffle=self.shuffle_data, random_state=self.shuffle_random_seed)
        num_samples = X.shape[0]
        split_indices = list(cv.split(X))
        indices = np.arange(0, num_samples, 1)

        # tuple of residuals from first stage estimation [0,1], and the confounders [2]
        residuals = self.estimator.residuals_
        residualized_outcome = residuals[0]  # T-E[T|W]
        residualized_treatment = residuals[1]  # Y - E[Y|W]
        W = residuals[3]

        n_residuals = residualized_outcome.shape[0]
        indices = np.arange(0, n_residuals, 1)

        residualized_outcome = residualized_outcome[indices]
        residualized_treatment = residualized_treatment[indices]

        # We need to estimate, sigma^2 = (Y-g_s)^2. We use the following derivation.
        # Yres = Y - E[Y|W]
        # E[Y|W] = f(x) + theta_s * E[T|W]
        # Yres = Y - f(x) - theta_s * E[T|W]
        # g(s) = theta_s * T + f(x)
        # g(s) = theta_s * (T - E[T|W]) + f(x) + theta_s * E[T|W]
        # g(s) = theta_s * Tres +f(x) + theta_s * E[T|W]
        # Y - g(s) = Y - [theta_s * Tres + f(x) + theta_s * E[T|W] )
        # Y - g(s) = ( Y - f(x) -  theta_s * E[T|W]) - theta_s * Tres
        # Y - g(s) = Yres - theta_s * Tres
        residualized_outcome_second_stage = residualized_outcome - self.theta_s * residualized_treatment
        self.sigma_2 = np.mean(residualized_outcome_second_stage**2)
        # nu_2 is E[alpha_s^2]
        self.nu_2 = np.mean(residualized_treatment**2)

        self.S2 = self.sigma_2 / self.nu_2

        # Now computing scores for finding the (1-a) confidence interval
        self.neyman_orthogonal_score_outcome = (
            residualized_outcome_second_stage * residualized_outcome_second_stage - self.sigma_2
        )
        self.neyman_orthogonal_score_treatment = residualized_treatment * residualized_treatment - self.nu_2
        self.neyman_orthogonal_score_theta = (residualized_outcome_second_stage) * residualized_treatment / self.nu_2

        # R^2 of treatment with observed common causes
        reg_function_fit = self.estimator.models_t[0][0]  # First Stage treatment model
        treatment_model = reg_function_fit.predict(W)
        self.r2t_w = np.var(treatment_model) / np.var(T)

        # R^2 of outcome with treatment and observed common causes
        self.g_s = Y - residualized_outcome_second_stage
        self.r2y_tw = np.var(self.g_s) / np.var(Y)

        self.g_s_j = np.zeros(num_samples)
        if self.benchmarking:
            delta_r2_y_wj, delta_r2t_wj = self.compute_r2diff_benchmarking_covariates(
                treatment_df,
                features,
                T,
                Y,
                W,
                self.benchmark_common_causes,
                split_indices=split_indices,
                second_stage_linear=False,
                is_partial_linear=self.is_partial_linear,
            )

            # Partial R^2 of outcome after regressing over unobserved confounder, observed common causes and treatment
            delta_r2y_u = self.frac_strength_outcome * delta_r2_y_wj
            # Partial R^2 of treatment after regressing over unobserved confounder and observed common causes
            delta_r2t_u = self.frac_strength_treatment * delta_r2t_wj
            self.r2yu_tw = delta_r2y_u / (1 - self.r2y_tw)
            self.r2tu_w = delta_r2t_u / (1 - self.r2t_w)
            if self.r2yu_tw >= 1:
                self.r2yu_tw = 1
                self.logger.warning(
                    "Warning: r2yu_tw can not be > 1. Try a lower effect_fraction_on_outcome. Setting r2yu_tw to 1"
                )
            if self.r2tu_w >= 1:
                self.r2tu_w = 1
                self.logger.warning(
                    "Warning: r2tu_w can not be > 1. Try a lower effect_fraction_on_treatment. Setting r2tu_w to 1"
                )
            if self.r2yu_tw < 0:
                self.r2yu_tw = 0
            if self.r2tu_w < 0:
                self.r2tu_w = 0
        else:
            self.r2yu_tw = self.effect_strength_outcome
            self.r2tu_w = self.effect_strength_treatment

        benchmarking_results = self.perform_benchmarking(
            r2yu_tw=self.r2yu_tw,
            r2tu_w=self.r2tu_w,
            significance_level=self.significance_level,
            is_partial_linear=self.is_partial_linear,
        )
        self.results = pd.DataFrame(benchmarking_results, index=[0])

        self.RV = self.calculate_robustness_value(alpha=None, is_partial_linear=self.is_partial_linear)
        self.RV_alpha = self.calculate_robustness_value(
            alpha=self.significance_level, is_partial_linear=self.is_partial_linear
        )

        if plot == True:
            self.plot()

        return self

    def plot(
        self,
        plot_type="lower_confidence_bound",
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
        legend_position=(1.05, 1),
    ):
        """
        Plots and summarizes the sensitivity bounds as a contour plot, as they vary with the partial R^2 of the unobserved confounder(s) with the treatment and the outcome
        Two types of plots can be generated, based on adjusted estimates or adjusted t-values
        X-axis: Partial R^2 of treatment and unobserved confounder(s)
        Y-axis: Partial R^2 of outcome and unobserved confounder(s)
        We also plot bounds on the partial R^2 of the unobserved confounders obtained from observed covariates

        :param plot_type: possible values are 'bias','lower_ate_bound','upper_ate_bound','lower_confidence_bound','upper_confidence_bound'
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
        :param unadjusted_estimate_color: marker color for unadjusted estimate in the plot (default = "black")
        :param adjusted_estimate_marker: marker type for bias adjusted estimates in the plot (default = '^')
        :parm adjusted_estimate_color: marker color for bias adjusted estimates in the plot (default = "red")
        :param legend_position:tuple denoting the position of the legend (default = (1.6, 0.6))
        """
        critical_value = 0

        fig, ax = plt.subplots(1, 1, figsize=plot_size)
        ax.set_title("Sensitivity contour plot of %s" % plot_type)
        if self.is_partial_linear:
            ax.set_xlabel("Partial R^2 of unobserved confounder with treatment")
        else:
            ax.set_xlabel("Fraction of the variance in Reisz function explained by unobserved confounder")
        ax.set_ylabel("Partial R^2 of unobserved confounder with outcome")
        if self.effect_strength_treatment is None:
            # adding 1.1 as plotting margin  ensure that the benchmarked part is shown fully in plot
            x_limit = (1.1 * self.r2tu_w) if self.benchmarking else 0.99
            r2tu_w = np.arange(0.0, x_limit, x_limit / self.num_points_per_contour)
        else:
            x_limit = max(self.r2tu_w)
            r2tu_w = self.r2tu_w
        if self.effect_strength_outcome is None:
            # adding 1.1 as plotting margin  ensure that the benchmarked part is shown fully in plot
            y_limit = (1.1 * self.r2yu_tw) if self.benchmarking else 0.99
            r2yu_tw = np.arange(0.0, y_limit, y_limit / self.num_points_per_contour)
        else:
            y_limit = self.r2yu_tw[-1]
            r2yu_tw = self.r2yu_tw
        ax.set_xlim(-x_limit / 20, x_limit)
        ax.set_ylim(-y_limit / 20, y_limit)

        undjusted_estimates = None
        contour_values = np.zeros((len(r2yu_tw), len(r2tu_w)))

        for i in range(len(r2yu_tw)):
            y = r2yu_tw[i]
            for j in range(len(r2tu_w)):
                x = r2tu_w[j]
                benchmarking_results = self.perform_benchmarking(
                    r2yu_tw=y,
                    r2tu_w=x,
                    significance_level=self.significance_level,
                    is_partial_linear=self.is_partial_linear,
                )
                contour_values[i][j] = benchmarking_results[plot_type]

        contour_plot = ax.contour(
            r2tu_w,
            r2yu_tw,
            contour_values,
            colors=contours_color,
            linewidths=contour_linewidths,
            linestyles=contour_linestyles,
        )
        ax.clabel(contour_plot, inline=1, fontsize=label_fontsize, colors=contours_label_color)

        if critical_value >= contour_values.min() and critical_value <= contour_values.max() and plot_type != "bias":
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
        if (
            plot_type == "lower_confidence_bound"
            or plot_type == "upper_confidence_bound"
            or plot_type == "lower_ate_bound"
            or plot_type == "upper_ate_bound"
        ):
            ax.scatter(
                [0],
                [0],
                marker=unadjusted_estimate_marker,
                color=unadjusted_estimate_color,
                label="Unadjusted({:1.2f})".format(self.theta_s),
            )

        # Adding bounds to partial R^2 values for given strength of confounders
        if self.benchmarking:
            if self.frac_strength_treatment == self.frac_strength_outcome:
                signs = str(round(self.frac_strength_treatment, 2))
            else:
                signs = str(round(self.frac_strength_treatment, 2)) + "/" + str(round(self.frac_strength_outcome, 2))
            label = signs + " X " + str(self.benchmark_common_causes) + " ({:1.2f}) ".format(self.results[plot_type][0])
            ax.scatter(
                self.r2tu_w, self.r2yu_tw, color=adjusted_estimate_color, marker=adjusted_estimate_marker, label=label
            )

        plt.margins()
        ax.legend(bbox_to_anchor=legend_position, loc="upper left")
        plt.show()

    def __str__(self):
        s = "Sensitivity Analysis to Unobserved Confounding using partial R^2 parameterization\n\n"
        s += "Original Effect Estimate : {0}\n".format(self.theta_s)
        s += "Robustness Value : {0}\n\n".format(self.RV)
        s += "Robustness Value (alpha={0}) : {1}\n\n".format(self.significance_level, self.RV_alpha)
        s += "Interpretation of results :\n"
        s += "Any confounder explaining less than {0}% percent of the residual variance of both the treatment and the outcome would not be strong enough to explain away the observed effect i.e bring down the estimate to 0 \n\n".format(
            round(self.RV * 100, 2)
        )
        s += "For a significance level of {0}%, any confounder explaining more than {1}% percent of the residual variance of both the treatment and the outcome would be strong enough to make the estimated effect not 'statistically significant'\n\n".format(
            self.significance_level * 100, round(self.RV_alpha * 100, 2)
        )
        return s
