import scipy
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from econml.utilities import cross_product
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from dowhy.utils.util import get_numeric_features
from dowhy.causal_refuters.partial_linear_sensitivity_analyzer import PartialLinearSensitivityAnalyzer
from dowhy.causal_refuters.reisz import ReiszRegressor, ReiszRepresenter, generate_moment_function, get_alpha_estimator, get_generic_regressor, create_polynomial_function


class NonParametricSensitivityAnalyzer(PartialLinearSensitivityAnalyzer):
    """
    Non-parametric sensitivity analysis for causal estimators. 

    Currently works only for single-dimensional treatment. 

    Two important quantities used to estimate the bias are alpha and g.
         g := E[Y | T, W, Z] denotes the long regression function
         g_s := E[Y | T, W] denotes the short regression function
         α := (T - E[T | W, Z] ) / (E(T - E[T | W, Z]) ^ 2) denotes long reisz representer
         α_s := (T - E[T | W] ) / (E(T - E[T | W]) ^ 2) denotes short reisz representer
    Bias = E(g_s - g)(α_s - α) 
    Thus, The bound is the product of additional variations that omitted confounders generate in the regression function and in the reisz representer for partially linear models.
    It can be written as,  Bias = S * Cg * Calpha 
    where Cg and Calpha are explanatory powers of the confounder and S^2 = E(Y - g_s) ^ 2 * E(α_s ^ 2)

    Based on this work:
        Chernozhukov, Victor, Carlos Cinelli, Whitney Newey, Amit Sharma, and Vasilis Syrgkanis. Long Story Short: Omitted Variable Bias in Causal Machine Learning. No. w30302. National Bureau of Economic Research, 2022.

    :param estimator: estimator of the causal model
    :param num_splits: number of splits for cross validation. (default = 5)
    :param shuffle_data : shuffle data or not before splitting into folds (default = False)
    :param shuffle_random_seed: seed for randomly shuffling data
    :param max_frac_inc_reisz: proportion of residual variance in the treatment explained by confounders
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
    :param theta_s: point estimate for the estimator
    
    """

    def __init__(self, *args, theta_s, 
            max_frac_inc_reisz = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_s = theta_s
        self.max_frac_inc_reisz = max_frac_inc_reisz
        
        # whether the DGP is assumed to be partially linear
        self.is_partial_linear = False

        self.moment_function = None    
        self.alpha_s = None
        self.m_alpha = None
        self.g_s = None
        self.m_g = None

    def check_sensitivity(self, plot=True):
        """
        Function to perform sensitivity analysis. 
        The following formulae are used to obtain the upper and lower bound respectively.
        θ+ = θ_s + S * C_g * C_α
        θ- = θ_s - S * C_g * C_α
        where θ_s is the obtained estimate,  S^2 = E[Y - gs]^2  * E[α_s]^ 2
        S is obtained by debiased machine learning.
        θ_s = E[m(W, gs) + (Y - gs) * α_s]
        σ² = E[Y - gs]^2
        ν^2 = 2 * E[m(W, α_s )] - E[α_s ^ 2]

        :param plot: plot = True generates a plot of lower confidence bound of the estimate for different variations of unobserved confounding.
                     plot = False overrides the setting
        
        :returns: instance of NonParametricSensitivityAnalyzer class
        """

        # features are the observed confounders
        features = self.observed_common_causes.copy()
        treatment_df = self.treatment.copy()
        W = features.to_numpy()
        X = pd.concat([treatment_df, features], axis=1)
        numeric_features = get_numeric_features(X = X)
        T = treatment_df.values.ravel()

        X = X.to_numpy()
        Y = self.outcome.copy()
        Y = Y.values.ravel()

        cv = KFold(n_splits=self.num_splits, shuffle=self.shuffle_data,
                   random_state=self.shuffle_random_seed)
        num_samples = X.shape[0]
        split_indices = list(cv.split(X))
        indices = np.arange(0, num_samples, 1)

        self.moment_function = generate_moment_function

        self.alpha_s = np.zeros(num_samples)
        self.m_alpha = np.zeros(num_samples)
        self.g_s = np.zeros(num_samples)
        self.m_g = np.zeros(num_samples)
        self.g_s_j = np.zeros(num_samples)
        self.alpha_s_j = np.zeros(num_samples)

        reg_function = get_generic_regressor(cv=split_indices,
                                             X=X, Y=Y, max_degree=self.reisz_polynomial_max_degree,
                                             estimator_list=self.g_s_estimator_list,
                                             estimator_param_list=self.g_s_estimator_param_list,
                                             numeric_features=numeric_features
                                             )

        reisz_function = get_alpha_estimator(
            cv=split_indices, X=X, max_degree=self.reisz_polynomial_max_degree, param_grid_dict=self.alpha_s_param_dict)

        for train, test in split_indices:
            reisz_fn_fit = reisz_function.fit(X[train])
            self.alpha_s[test] = reisz_fn_fit.predict(X[test])
            self.m_alpha[test] = self.moment_function(
                X[test], reisz_fn_fit.predict)

            reg_fn_fit = reg_function.fit(X[train], Y[train])
            self.g_s[test] = reg_fn_fit.predict(X[test])
            self.m_g[test] = self.moment_function(X[test], reg_fn_fit.predict)

        self.m = self.m_g + self.alpha_s * (Y - self.g_s)

        self.nu_2 = np.mean(
            2 * self.m_alpha[indices] - self.alpha_s[indices] ** 2)
        self.sigma_2 = np.mean((Y[indices] - self.g_s[indices]) ** 2)
        self.S2 = self.nu_2 * self.sigma_2
        self.S = np.sqrt(self.S2)

        Y_residual = Y[indices] - self.g_s[indices]
        self.neyman_orthogonal_score_outcome = Y_residual ** 2 - self.sigma_2
        self.neyman_orthogonal_score_treatment = 2 * \
            self.m_alpha[indices] - self.alpha_s[indices] ** 2 - self.nu_2
        self.neyman_orthogonal_score_theta = self.m[indices] - self.theta_s

        # Now code for benchmarking using covariates begins
        # R^2 of outcome with observed common causes and treatment
        self.r2y_tw = np.var(self.g_s) / np.var(Y)

        # R^2 of treatment with observed common causes
        numeric_features = get_numeric_features(features)
        self.r2t_w = self.get_regression_r2(X = W, Y = T, numeric_features = numeric_features, split_indices = split_indices)

        delta_r2_y_wj, var_alpha_wj = self.compute_r2diff_benchmarking_covariates(
            treatment_df, features, T, Y, W, self.benchmark_common_causes,
            split_indices=split_indices, is_partial_linear=self.is_partial_linear)

        # Partial R^2 of outcome after regressing over unobserved confounder, observed common causes and treatment
        delta_r2y_u = self.frac_strength_outcome * delta_r2_y_wj
        # The difference in R2 is the same for wj and new unobserved confounder 
        c2_alpha = self.frac_strength_treatment * (np.var(self.alpha_s)/var_alpha_wj - 1)#self.frac_strength_treatment * delta_r2t_wj

        self.r2yu_tw = abs(delta_r2y_u / (1 - self.r2y_tw))
        self.r2tu_w = c2_alpha
        if self.r2yu_tw >= 1:
            self.r2yu_tw = 1
            self.logger.warning("Warning: r2yu_tw can not be > 1. Try a lower effect_fraction_on_outcome. Setting r2yu_tw to 1")

        benchmarking_results = self.perform_benchmarking(
            r2yu_tw=self.r2yu_tw, r2tu_w=self.r2tu_w,
            significance_level=self.significance_level,
            is_partial_linear=self.is_partial_linear)
        self.results = pd.DataFrame(benchmarking_results, index=[0])

        self.RV = self.calculate_robustness_value(alpha=None, is_partial_linear=self.is_partial_linear)
        self.RV_alpha = self.calculate_robustness_value(
            alpha=self.significance_level,
            is_partial_linear=self.is_partial_linear)
        if plot == True:
            self.plot(x_limit=self.max_frac_inc_reisz)

        return self

    def get_phi_lower_upper(self, Cg, Calpha):
        """
        Calculate lower and upper influence function (phi)

        :param Cg: measure of strength of confounding that omitted variables generate in outcome regression
        :param Calpha: measure of strength of confounding that omitted variables generate in treatment regression

        :returns : lower bound of phi, upper bound of phi
        """

        bounds_estimator_upper = self.neyman_orthogonal_score_theta + ((Cg * Calpha) / (2 * self.S)) * (
            self.sigma_2 * self.neyman_orthogonal_score_treatment + self.nu_2 * self.neyman_orthogonal_score_treatment)
        bounds_estimator_lower = self.neyman_orthogonal_score_theta - ((Cg * Calpha) / (2 * self.S)) * (
            self.sigma_2 * self.neyman_orthogonal_score_treatment + self.nu_2 * self.neyman_orthogonal_score_treatment)

        return bounds_estimator_lower, bounds_estimator_upper

    def get_alpharegression_var(self, X, split_indices,
            reisz_model=None):
        """
        Calculates the pearson non parametric partial R^2 from a regression function.

        :param X: numpy array containing set of regressors 
        :param numeric_features: list of indices of columns with numeric features
        :param split_indices: training and testing data indices obtained after cross folding

        :returns: partial R^2 value
        """
        if reisz_model is None:
            reisz_model = get_alpha_estimator(
                cv=split_indices, X=X, 
                max_degree=self.reisz_polynomial_max_degree, 
                param_grid_dict=self.alpha_s_param_dict)

        num_samples = X.shape[0]
        pred = np.zeros(num_samples) 
        for train, test in split_indices:
            reisz_fn_fit = reisz_model.fit(X[train])
            pred[test] = reisz_fn_fit.predict(X[test])

        return np.var(pred)


