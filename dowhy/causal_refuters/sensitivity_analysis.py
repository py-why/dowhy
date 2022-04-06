import statsmodels.formula.api as smfa
import numpy as np
from scipy.stats import t
import pandas as pd
import statsmodels.api as sma
import logging
import matplotlib.pyplot as plt


class Sensitivity_analysis:
    """
    Class to perform sensitivity analysis
    :param formula: The formula specifying the model
    :param data: Pandas dataframe
    :param treatment_name: name of treatment
    :param q: percentage reduction for robustness value
    :param standard_error: standard error in regression
    :param estimate: estimate of regression
    :param degree_of_freedom: degree of freedom of error in regression
    :param h0: hypothesis
    :param stats: dictionary for sensitivity statistics
    :param benchmark_covariates: names of variables for benchmark bounding
    :param kd: strength of association between benchmark and treatment variable to test with benchmark bounding
    :param ky: strength of association between benchmark and outcome variable to test with benchmark bounding
    :param r2dz_x: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
    :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
    :param r2dxj_x: partial R^2 of covariate Xj with treatment "d", covariates "x" excluding Xj are partialed out
    :param r2yxj_dx:  partial R^2 of covariate Xj with outcome "y", covariates "x" excluding Xj and treatment "d" are partialed out
    :param bounds_results: dataframe containing information about bounds and bias adjusted terms
    :param stats: dictionary containing information like robustness value, partial R^2, estimate, standard error , degree of freedom, partial f^2, t-statistic
    """
    formula = None
    data = None
    treatment_name = None
    estimate = None
    degree_of_freedom = None
    standard_error = None
    q = None
    confidence = None
    h0 = 0
    increase = False
    benchmark_covariates = None
    kd = None
    ky = None
    r2dz_x = None
    r2yz_dx = None
    r2dxj_x = None
    r2yxj_dx = None
    bias_adjusted_estimate = None
    bias_adjusted_se = None
    bias_adjusted_t = None
    bias_adjusted_lower_CI = None
    bias_adjusted_upper_CI = None
    bounds_result = None
    stats = None
    

    def __init__(self, formula = None, data = None, treatment_name = None, q = 1.0, confidence = 0.05, increase = False, benchmark_covariates = None, kd = None, ky = None):
        self.formula = formula
        self.data = data
        self.treatment_name = treatment_name
        self.q = q
        self.confidence = confidence
        self.increase = increase
        self.benchmark_covariates = benchmark_covariates
        self.h0 = 0
        if type(kd) is list:
            self.kd = np.array(kd)
        if type(ky) is list:
            self.ky = np.array(ky)
        self.logger = logging.getLogger(__name__)
    
    def get_OLS_results(self, model, covariates):
        """
        Computes result parameters of the OLS model

        :param model: OLS regression model
        :param covariates: list of covariates

        :returns: python dictionary with keys as covariates, standard_error, t_stats, degree_of_freedom, estimate 
        """
        model_details = {
            'covariates': covariates,
            'standard_error': model.bse[covariates],
            't_stats': model.tvalues[covariates],
            'degree_of_freedom': int(model.df_resid),
            'estimate': model.params[covariates]
        }
        return model_details
        
        
    def partial_r2_func(self, model = None, treatment = None):
        """
        Computes the partial R^2 of regression model 

        :param model: statsmodels OLS regression model
        :param treatment: treatment name

        :returns: partial R^2 value
        """
        model_details = self.get_OLS_results(model, treatment)
        t_stats = model_details['t_stats']
        degree_of_freedom = model_details['degree_of_freedom']
        return  (t_stats ** 2 / (t_stats ** 2 + degree_of_freedom))
    

    def group_partial_r2_func(self, model = None, treatment = None):
        """
        Compute partial r2 of a group of covariates

        :param model: OLS regression model
        :param treatment: treatment name

        :returns: partial R^2 value of group of covariates
        """
        model_details = self.get_OLS_results(model, treatment)
        estimate = model_details['estimate']
        if np.isscalar(estimate):
            return self.partial_r2_func(model, treatment)
        degree_of_freedom = model_details['degree_of_freedom']
        v = model.cov.params().loc[treatment, :][treatment]
        l = len(estimate)
        f_stats = np.matmul(np.matmul(estimate.values.T, np.linalg.inv(v.values)), estimate.values) / l
        return f_stats * l / (f_stats * p + degree_of_freedom)


    def get_all_covariates(self, treatment):
        """
        Converts a single treatment variable to an array

        :param treatment: name(s) of the treatment
        :returns : python array
        """
        if type(treatment) is str:
            treatment = [treatment]
        return treatment


        
    def perform_analysis(self):
        """
        Function to perform sensitivity analysis

        """
        regression_model = smfa.ols(formula = self.formula, data = self.data)
        results = regression_model.fit()
        if int(results.df_resid) == 0:
            raise ValueError("Zero residual degrees of freedom")

        if self.treatment_name is not None:
            covariates = self.get_all_covariates(self.treatment_name)
        else:
            covariates = results.model.exog_names

        self.standard_error = list(results.bse[covariates])[0]
        self.degree_of_freedom = int(results.df_resid)
        self.estimate = list(results.params[covariates])[0]

        if self.increase:
            self.h0 = self.estimate * (1 + self.q)
        else:
            self.h0 = self.estimate * (1 - self.q)
        t_value = self.estimate / self.standard_error
        partial_r2 = self.partial_r2_func(results, covariates)
        rv_q = self.robustness_value(model = results, t_statistic = t_value)
        rv_q_alpha = self.robustness_value(model = results,t_statistic = t_value, alpha = self.confidence)
        partial_f2 = t_value ** 2 / self.degree_of_freedom
        t_statistic = (self.estimate - self.h0)/ self.standard_error
        self.stats = {'estimate' : self.estimate,
        'standard error' : self.standard_error,
        'degree of freedom' : self.degree_of_freedom,
        't_statistic' : t_statistic,
        'r2yd_x' : partial_r2,
        'partial_f2' : partial_f2,
        'robustness_value' : rv_q,
        'robustness_value_alpha ' : rv_q_alpha
        }

        m = pd.DataFrame(results.model.exog, columns = results.model.exog_names)
        d = np.array(m[self.treatment_name])
        nd = m.drop(columns = self.treatment_name)
        nd.insert(0,0,1)
        model = sma.OLS(d, nd)
        treatment_results = model.fit()
        if type(self.benchmark_covariates) is str:
            self.r2yxj_dx = self.partial_r2_func(results, self.benchmark_covariates)
            self.r2dxj_x = self.partial_r2_func(treatment_results, self.benchmark_covariates)
        else:
            self.r2dxj_x = []
            self.r2yxj_dx = []
            for covariate in self.benchmark_covariates:
                self.r2yxj_dx.append(self.group_partial_r2_func(results, covariate))
                self.r2dxj_x.append(self.group_partial_r2_func(treatment_results, covariate))
        bounds = pd.DataFrame()
        for i in range(len(self.benchmark_covariates)):
            r2dxj_x = self.r2dxj_x[i]
            r2yxj_dx = self.r2yxj_dx[i]
            if self.ky is None:
                self.ky = self.kd
            self.r2dz_x = self.kd * (r2dxj_x / (1 - r2dxj_x))
            r2zxj_xd = self.kd * (r2dxj_x ** 2) / ((1 - self.kd * r2dxj_x) * (1 - r2dxj_x))
            self.r2yz_dx = ((np.sqrt(self.ky) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd)) ** 2 * (r2yxj_dx / (1 - r2yxj_dx))
            if np.isscalar(self.kd):
                bounds = bounds.append({'r2dz_x' : self.r2dz_x, 'r2yz_dx' : self.r2yz_dx}, ignore_index = True)
            else:
                for k in range(len(self.kd)):
                    bounds = bounds.append({'r2dz_x' : self.r2dz_x[k], 'r2yz_dx' : self.r2yz_dx[k]}, ignore_index = True)

        #Calculate bias adjusted terms
        self.compute_bias_adjusted()
        if np.isscalar(self.r2dz_x):
            self.bounds_result = pd.DataFrame(data = {
                'r2dz_x': self.r2dz_x,
                'r2yz_dx': self.r2yz_dx,
                'treatment': self.treatment_name * len(self.kd),
                'adjusted_estimate': self.bias_adjusted_estimate,
                'adjusted_se': self.bias_adjusted_se,
                'adjusted_t': self.bias_adjusted_t,
                'adjusted_lower_CI': self.bias_adjusted_lower_CI,
                'adjusted_upper_CI': self.bias_adjusted_upper_CI
            }, index = [0])
        else:
            self.bounds_result = pd.DataFrame(data = {
                'r2dz_x': self.r2dz_x,
                'r2yz_dx': self.r2yz_dx,
                'treatment': self.treatment_name * len(self.kd),
                'adjusted_estimate': self.bias_adjusted_estimate,
                'adjusted_se': self.bias_adjusted_se,
                'adjusted_t': self.bias_adjusted_t,
                'adjusted_lower_CI': self.bias_adjusted_lower_CI,
                'adjusted_upper_CI': self.bias_adjusted_upper_CI
            })

        self.plot()
        return self


    def robustness_value(self, model = None, covariates = None, t_statistic = None, alpha = 1.0):
        """
        Function to calculate the robustness value. 
        It is the minimum strength of association that omitted variables must have with treatment and outcome to change the estimated coefficient by certain amount

        :param model: OLS regression model
        :param covariates: names of covariates
        :param t_statistic: t- value of the OLS regression
        :param alpha: confidence interval (default = 1)

        :returns: robustness value 
        """
        t_statistic = list(self.get_OLS_results(model = model, covariates = self.treatment_name)['t_stats'])[0]
        cohen_f = self.q * abs(t_statistic / np.sqrt(self.degree_of_freedom))
        f_critical = abs(t.ppf(alpha / 2 , self.degree_of_freedom - 1)) / np.sqrt(self.degree_of_freedom - 1)
        f_q_alpha = cohen_f - f_critical

        if(type(t_statistic) is float or type(t_statistic) is int):
            t_statistic = pd.Series(t_statistic)

        rv = 0.5 * (np.sqrt(f_q_alpha ** 4 + (4 * f_q_alpha ** 2)) - f_q_alpha ** 2)
        rvx = (cohen_f ** 2 - f_critical ** 2) / (1 + cohen_f ** 2)
        if f_q_alpha < 0:
            rv_out = 0
        else:
            rv_out = rv
        if f_q_alpha > 0 and cohen_f > 1 / f_critical:
            rv_out = rvx
        return rv_out
    
    
    def compute_bias_adjusted(self):
        """
        Computes the bias adjusted estimate, standard error, t-value,  partial R2, confidence intervals
        """
        
        r2dz_x = np.array(self.r2dz_x)
        r2yz_dx = np.array(self.r2yz_dx)

        bias = self.bias(self.r2dz_x, self.r2yz_dx, self.standard_error, self.degree_of_freedom)
        
        self.bias_adjusted_estimate = self.bias_estimate(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx, estimate = self.estimate)
        
        self.bias_adjusted_se = self.bias_se(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx)

        self.bias_adjusted_t = self.bias_t(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx, h0 = self.h0, estimate = self.estimate)

        self.bias_adjusted_partial_r2 = self.bias_adjusted_t ** 2 / (self.bias_adjusted_t ** 2 + (self.degree_of_freedom - 1)) #partial r2 formula used with new t value and dof - 1

        num_se = t.ppf(self.confidence / 2, self.degree_of_freedom) # Number of standard errors within Confidence Interval

        self.bias_adjusted_upper_CI = self.bias_adjusted_estimate - num_se * self.bias_adjusted_se
        self.bias_adjusted_lower_CI = self.bias_adjusted_estimate + num_se * self.bias_adjusted_se

    def bias(self,  r2dz_x , r2yz_dx, standard_error, degree_of_freedom):
        """
        Calculate the bias
        :param r2dz_x: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param standard_error: standard error in regression
        :param degree_of_freedom: degree of freedom of error in regression

        :returns: bias
        """
        bias_factor = np.sqrt((r2yz_dx * r2dz_x) / (1 - r2dz_x))
        bias = bias_factor * (standard_error * np.sqrt(degree_of_freedom))
        return bias


    def bias_estimate(self, r2dz_x , r2yz_dx, estimate):
        """
        Calculates the bias adjusted estimate
        :param r2dz_x: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param estimate: estimate of regression

        :returns: the bias adjusted estimate
        """
        bias = self.bias(r2dz_x, r2yz_dx, self.standard_error, self.degree_of_freedom)
        if self.increase:
            return np.sign(estimate) * (abs(estimate) + bias)
        else:
            return np.sign(estimate) * (abs(estimate) - bias)


    def bias_se(self, r2dz_x, r2yz_dx ):
        """
        Calculates the bias adjusted standard error
        :param r2dz_x: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out

        :returns:  the bias adjusted standard error
        """
        return np.sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * self.standard_error * np.sqrt(self.degree_of_freedom / (self.degree_of_freedom - 1))


    def bias_t(self, r2dz_x, r2yz_dx, h0, estimate):
        """
        Calculates the bias adjusted t-value
        :param r2dz_x: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param h0: hypothesis value
        :param estimate: estimate of regression

        :returns:  the bias adjusted t-value
        """
        new_estimate = self.bias_estimate(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx, estimate = estimate)
        new_se = self.bias_se(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx)
        return (new_estimate - h0) / new_se

        
    def plot(self, plot_type = "contour", sensitivity_variable = 'estimate'):
        """
        Generate sensitivity plots of the analysis
        :param plot_type: "contour" by default. Might be extended later
        :param sensitivity_variable: "estimate" or "t-value". (default = 'estimate')
        """
        if plot_type == "contour":
            self.contour_plot(sensitivity_variable = sensitivity_variable )
        else:
            self.logger.error("Plot type not supported. Supported method : 'contour' ")

    def add_bound(self,bound_value, sensitivity_variable, label_names, x_limit, y_limit):
        """
        Function to add bound to the graph
        """
        if sensitivity_variable == 'estimate':
            bound_value = self.bounds_result['adjusted_estimate'].copy()
        else:
            bound_value = self.bounds_result['adjusted_t'].copy()
        
        for i in range(len(self.r2dz_x)):
            plt.scatter(self.r2dz_x[i], self.r2yz_dx[i], color = 'red', marker = 'D', edgecolors = 'black')
            plt.annotate(label_names[i], (self.r2dz_x[i] + x_limit / 40.0 , self.r2yz_dx[i] + y_limit / 40.0))

    def generate_label(self, benchmark_covariate):
        """
        Generate label names consisting of covariate name and appending multipliers ky and kd
        """
        label_names = []
        for i in range(len(self.kd)):
            kd = self.kd[i]
            ky = self.ky[i]
            if kd == ky:
                mult_text = str(round(ky,2))
            else:
                mult_text = str(round(kd,2)) + '/' + str(round(ky,2))
            label = mult_text + 'x' + ' ' + str(benchmark_covariate)
            label_names.append(label)
        return label_names


    def contour_plot(self,sensitivity_variable, kd = 1,  estimate_threshold = 0, t_threshold = 2, contour_color = "black", threshold_color = "red", x_limit = 0.4, y_limit = 0.4):
        """
        The horizontal axis shows hypothetical values of the patial R2 of unobserved confounder(s) with the treatment
        The vertical axis shows hypothetical values of the patial R2 of unobserved confounder(s) with the outcome.
        The contour levels represent adjusted t-values or estimates of unobserved confounding.

        :param sensitivity_variable: "estimate" or "t-value"
        :param kd: strength of association between benchmark and treatment variable to test with benchmark bounding
        :param r2yz_dx: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param estimate_threshold: threshold line to emphasize when contours correspond to estimate, (default = 0)
        :param t_threshold: threshold line to emphasize when contours correspond to t-value, (default = 2)
        :param x_limit: x axis maximum
        :param y_limit: y axis maximum
        :param contour_color: color of contour line(default = black)
        :param threshold_color: color of threshold line(default = red)
        """

        if self.increase:
            estimate_threshold = self.estimate * (1 + self.q)
        else:
            estimate_threshold = self.estimate * (1 - self.q)
        
        t_threshold = abs(t.ppf(self.confidence / 2, self.degree_of_freedom - 1)) * np.sign(self.stats['t_statistic'])
        
        grid_x = np.arange(0, x_limit, x_limit/400)
        grid_y = np.arange(0, y_limit, y_limit/400)

        bound_value = None

        if sensitivity_variable == 'estimate':
            z_axis = [[self.bias_estimate(grid_x[j], grid_y[i], self.estimate) for j in range(len(grid_x))] for i in range(len(grid_y))]
            threshold = estimate_threshold
            plot_estimate = self.estimate
            bound_value = self.bias_estimate(self.r2dz_x, self.r2yz_dx, self.estimate)
        else:
            z_axis = [[self.bias_t(grid_x[j], grid_y[i], estimate_threshold, self.estimate) for j in range(len(grid_x))] for i in range(len(grid_y))]
            threshold = t_threshold
            plot_estimate = (self.estimate - estimate_threshold) / self.standard_error
            bound_value = self.bias_t(self.r2dz_x, self.r2yz_dx, estimate_threshold, self.estimate)
        
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        cs = ax.contour(grid_x, grid_y, z_axis, colors = contour_color, linewidths = 1.0, linestyles = "solid")
        
        ax.clabel(cs, inline = 1, fontsize = 8, colors = "gray")
        cs = ax.contour(grid_x, grid_y, z_axis, colors = threshold_color, linewidths = 1.0, linestyles = [(0, (7, 3))], levels = [threshold])
        ax.clabel(cs, inline = 1, fontsize = 8, colors = "gray")

        ax.scatter([0], [0], c = 'k', marker = '^')
        ax.annotate("Unadjusted\n({:1.3f})".format(plot_estimate), (0.0, 0.0))

        plt.xlabel(r"Partial $R^2$ of confounder(s) with the treatment")
        plt.ylabel(r"Partial $R^2$ of confounder(s) with the outcome")
        plt.xlim(-(x_limit / 15.0), x_limit)
        plt.ylim(-(y_limit / 15.0), y_limit)

        label_names = self.generate_label(self.benchmark_covariates)
        
        if self.r2dz_x is not None:
            self.add_bound(bound_value, sensitivity_variable, label_names, x_limit, y_limit)
        plt.tight_layout()