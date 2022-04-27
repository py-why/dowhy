import statsmodels.formula.api as smfa
import numpy as np
from scipy.stats import t
import pandas as pd
import statsmodels.api as sma
import logging
import matplotlib.pyplot as plt
from dowhy.utils.api import parse_state
import sys

class LinearSensitivityAnalysis:
    """
    Class to perform sensitivity analysis
    See: https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20(2020)%20-%20Making%20Sense%20of%20Sensitivity.pdf 

    :param model: OLS results derived from linear estimator of the causal model
    :param data: Pandas dataframe
    :param treatment_name: name of treatment
    :param percent_change_rvalue: percentage reduction for robustness value
    :param h0: hypothesis
    :param stats: dictionary for sensitivity statistics
    :param benchmark_covariates: names of variables for benchmark bounding
    :param significance_level: confidence interval for statistical inference(default = 0.05)
    :param frac_strength_treatment: strength of association between benchmark and treatment variable to test with benchmark bounding
    :param frac_strength_outcome: strength of association between benchmark and outcome variable to test with benchmark bounding
    :param common_causes_order: The order of column names in OLS regression data
    """
    

    def __init__(self, OLSmodel = None, data = None, treatment_name = None, percent_change_rvalue = 1.0, significance_level = 0.05, increase = False, benchmark_covariates = None, frac_strength_treatment = None, frac_strength_outcome = None, common_causes_order = None):
        self.data = data
        self.treatment_name = []
        # original_treatment_name: : stores original variable names for labelling
        self.original_treatment_name = treatment_name
        for i in range(len(treatment_name)):
            self.treatment_name.append("x"+str(i+1))
        self.percent_change_rvalue = percent_change_rvalue
        self.significance_level = significance_level
        self.increase = increase
        self.OLSmodel = OLSmodel
        self.h0 = 0
        # common_causes_map : maps the original variable names to variable names in OLS regression
        self.common_causes_map = {}
        for i in range(len(common_causes_order)):
            self.common_causes_map[common_causes_order[i]] = "x"+str(len(self.treatment_name)+i+1)
        # benchmark_covariates: stores variable names in terms of regression model variables
        self.benchmark_covariates = []
        # original_benchmark_covariates: stores original variable names for labelling
        self.original_benchmark_covariates = benchmark_covariates
        for i in range(len(benchmark_covariates)):
            self.benchmark_covariates.append(self.common_causes_map[benchmark_covariates[i]])
        if type(frac_strength_treatment) is list:
            self.frac_strength_treatment = np.array(frac_strength_treatment)
        if type(frac_strength_outcome) is list:
            self.frac_strength_outcome = np.array(frac_strength_outcome)
        treatment_name = None
        # estimate: estimate of regression
        self.estimate = None
        # degree_of_freedom: degree of freedom of error in regression
        self.degree_of_freedom = None
        # standard_error: standard error in regression
        self.standard_error = None
        # r2tu_w: partial R^2  of putative unobserved confounder "u" with treatment "t", observed covariates "w" partialed out
        self.r2tu_w = None
        # r2yu_tw: partial R^2  of putative unobserved confounder "u" with outcome "y", observed covariates "w" and treatment "t" partialed out
        self.r2yu_tw = None
        # r2twj_w: partial R^2 of covariate Wj with treatment "t", covariates "w" excluding Wj are partialed out
        self.r2twj_w = None
        # r2ywj_tw:  partial R^2 of covariate Wj with outcome "y", covariates "w" excluding Wj and treatment "t" are partialed out
        self.r2ywj_tw = None
        self.bias_adjusted_estimate = None
        self.bias_adjusted_se = None
        self.bias_adjusted_t = None
        self.bias_adjusted_lower_CI = None
        self.bias_adjusted_upper_CI = None
        # bounds_results: dataframe containing information about bounds and bias adjusted terms
        self.bounds_result = None
        # stats: dictionary containing information like robustness value, partial R^2, estimate, standard error , degree of freedom, partial f^2, t-statistic
        self.stats = None
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
        

    def check_r2(self, partialR2_1, partialR2_2):
        """"
        Function to check if the given partial R^2 values are valid or not. 
        A valid partial R^2 value is between 0 and 1.
        """
        if(np.isscalar(partialR2_1) and partialR2_1 >= 1) or (not np.isscalar(partialR2_1) and any(val >= 1 for val in partialR2_1)):
            self.logger.error('Partial R^2 value should be between 0 and 1')
        if(np.isscalar(partialR2_2) and partialR2_2 >= 1) or (not np.isscalar(partialR2_2) and any(val >= 1 for val in partialR2_2)):
            self.logger.error('Partial R^2 value should be between 0 and 1')
        
            
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
        v = model.cov_params().loc[treatment, :][treatment]
        l = len(estimate) #Number of parameters in the model
        f_stats = np.matmul(np.matmul(estimate.values.T, np.linalg.inv(v.values)), estimate.values) / l
        return f_stats * l / (f_stats * l + degree_of_freedom)

        
    def perform_analysis(self):
        """
        Function to perform sensitivity analysis. 
        By default it generates a plot of point estimate and the variations with respect to benchmarking and unobserved confounding.
        """

        if self.treatment_name is not None:
            self.treatment_name = parse_state(state = self.treatment_name)
        else:
            self.treatment_name = self.OLSmodel.model.exog_names

        self.standard_error = list(self.OLSmodel.bse[1:(len(self.treatment_name)+1)])[0]
        self.degree_of_freedom = int(self.OLSmodel.df_resid)
        self.estimate = list(self.OLSmodel.params[1:(len(self.treatment_name)+1)])[0]

        if self.increase:
            self.h0 = self.estimate * (1 + self.percent_change_rvalue)
        else:
            self.h0 = self.estimate * (1 - self.percent_change_rvalue)
        t_value = self.estimate / self.standard_error
        partial_r2 = self.partial_r2_func(self.OLSmodel, self.treatment_name)
        rv_q = self.robustness_value(model = self.OLSmodel, t_statistic = t_value)
        rv_q_alpha = self.robustness_value(model = self.OLSmodel,t_statistic = t_value, alpha = self.significance_level)
        partial_f2 = t_value ** 2 / self.degree_of_freedom
        t_statistic = (self.estimate - self.h0)/ self.standard_error
        self.stats = {'estimate' : self.estimate,
        'standard error' : self.standard_error,
        'degree of freedom' : self.degree_of_freedom,
        't_statistic' : t_statistic,
        'r2yd_x' : list(partial_r2)[0],
        'partial_f2' : partial_f2,
        'robustness_value' : rv_q,
        'robustness_value_alpha' : rv_q_alpha
        }

        # build a new regression model by considering treatment variables as outcome 
        # r2twj_w is partial R^2 of covariate xj with treatment 
        m = pd.DataFrame(self.OLSmodel.model.exog, columns = self.OLSmodel.model.exog_names)
        d = np.array(m[self.treatment_name])  #Treatment 
        nd = m.drop(columns = self.treatment_name) #Non treatment
        nd.insert(0,0,1)
        model = sma.OLS(d, nd)
        treatment_results = model.fit()
        if type(self.benchmark_covariates) is str:
            self.r2ywj_tw = self.partial_r2_func(self.OLSmodel, self.benchmark_covariates) # partial R^2 of covariate Xj with outcome "y", covariates "x" excluding Xj and treatment "d" are partialed out
            self.r2twj_w = self.partial_r2_func(treatment_results, self.benchmark_covariates) # partial R^2 of covariate Xj with treatment "d", covariates "x" excluding Xj are partialed out
        else:
            self.r2twj_w = []
            self.r2ywj_tw = []
            for covariate in self.benchmark_covariates:
                self.r2ywj_tw.append(self.group_partial_r2_func(self.OLSmodel, covariate)) # partial R^2 of covariate Xj with outcome "y", covariates "x" excluding Xj and treatment "d" are partialed out
                self.r2twj_w.append(self.group_partial_r2_func(treatment_results, covariate)) # partial R^2 of covariate Xj with treatment "d", covariates "x" excluding Xj are partialed out
        bounds = pd.DataFrame()
        for i in range(len(self.benchmark_covariates)):
            r2twj_w = self.r2twj_w[i]
            r2ywj_tw = self.r2ywj_tw[i]
            if self.frac_strength_outcome is None:
                self.frac_strength_outcome = self.frac_strength_treatment
            self.r2tu_w = self.frac_strength_treatment * (r2twj_w / (1 - r2twj_w))
            if(np.isscalar(self.r2tu_w) and self.r2tu_w >= 1) or (not np.isscalar(self.r2tu_w) and any(val >= 1 for val in self.r2tu_w)):
                sys.exit("Implied bound on r2tu_w >= 1. Try a lower frac_strength_treatment")
            r2uwj_wt = self.frac_strength_treatment * (r2twj_w ** 2) / ((1 - self.frac_strength_treatment * r2twj_w) * (1 - r2twj_w))
            if(np.isscalar(r2uwj_wt) and r2uwj_wt >= 1) or (not np.isscalar(r2uwj_wt) and any(val >= 1 for val in r2uwj_wt)):
                sys.exit("Implied bound on r2uwj_wt >= 1. Try a lower frac_strength_treatment")
            self.r2yu_tw = ((np.sqrt(self.frac_strength_outcome) + np.sqrt(r2uwj_wt)) / np.sqrt(1 - r2uwj_wt)) ** 2 * (r2ywj_tw / (1 - r2ywj_tw))
            if(np.isscalar(self.r2yu_tw) and self.r2yu_tw >= 1) or (not np.isscalar(self.r2yu_tw) and any(val >= 1 for val in self.r2yu_tw)):
                for i in range(len(self.r2yu_tw)):
                    if self.r2yu_tw[i]>=1:
                        self.r2yu_tw[i]=1
                self.logger.warning("Warning: Implied bound on r2yu_tw >= 1. Try a lower frac_strength_treatment. Setting r2yu_tw to 1")
            if np.isscalar(self.frac_strength_treatment):
                bounds = bounds.append({'r2tu_w' : self.r2tu_w, 'r2yu_tw' : self.r2yu_tw}, ignore_index = True)
            else:
                for k in range(len(self.frac_strength_treatment)):
                    bounds = bounds.append({'r2tu_w' : self.r2tu_w[k], 'r2yu_tw' : self.r2yu_tw[k]}, ignore_index = True)

        #Calculate bias adjusted terms
        self.compute_bias_adjusted()
        
        self.bounds_result = pd.DataFrame(data = {
                'r2tu_w': self.r2tu_w,
                'r2yu_tw': self.r2yu_tw,
                'treatment': self.original_treatment_name * len(self.frac_strength_treatment),
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
        It is the minimum strength of association that omitted variables must have with treatment and outcome to change the estimated coefficient by certain amount.
        RVq describes how strong the association must be in order to reduce the estimated effect by (100 * percent_change_rvalue)%.
        RVq close to 1 means the treatment effect can handle strong confounders explaining  almost all residual variation of the treatment and the outcome.
        RVq close to 0 means that even very weak confounders could eliminate the results.

        :param model: OLS regression model
        :param covariates: names of covariates
        :param t_statistic: t- value of the OLS regression
        :param alpha: confidence interval (default = 1)

        :returns: robustness value 
        """
        t_statistic = list(self.get_OLS_results(model = model, covariates = self.treatment_name)['t_stats'])[0]
        cohen_f = self.percent_change_rvalue * abs(t_statistic / np.sqrt(self.degree_of_freedom))
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
        
        r2tu_w = np.array(self.r2tu_w)
        r2yu_tw = np.array(self.r2yu_tw)

        self.check_r2(r2tu_w, r2yu_tw)

        bias = self.bias(self.r2tu_w, self.r2yu_tw, self.standard_error, self.degree_of_freedom)
        
        self.bias_adjusted_estimate = self.bias_estimate(r2tu_w = r2tu_w, r2yu_tw = r2yu_tw, estimate = self.estimate)
        
        self.bias_adjusted_se = self.bias_se(r2tu_w = r2tu_w, r2yu_tw = r2yu_tw)

        self.bias_adjusted_t = self.bias_t(r2tu_w = r2tu_w, r2yu_tw = r2yu_tw, h0 = self.h0, estimate = self.estimate)

        self.bias_adjusted_partial_r2 = self.bias_adjusted_t ** 2 / (self.bias_adjusted_t ** 2 + (self.degree_of_freedom - 1)) #partial r2 formula used with new t value and dof - 1

        num_se = t.ppf(self.significance_level / 2, self.degree_of_freedom) # Number of standard errors within Confidence Interval

        self.bias_adjusted_upper_CI = self.bias_adjusted_estimate - num_se * self.bias_adjusted_se
        self.bias_adjusted_lower_CI = self.bias_adjusted_estimate + num_se * self.bias_adjusted_se

    def bias(self,  r2tu_w , r2yu_tw, standard_error, degree_of_freedom):
        """
        Calculate the bias
        :param r2tu_w: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yu_tw: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param standard_error: standard error in regression
        :param degree_of_freedom: degree of freedom of error in regression

        :returns: bias
        """
        bias_factor = np.sqrt((r2yu_tw * r2tu_w) / (1 - r2tu_w))
        bias = bias_factor * (standard_error * np.sqrt(degree_of_freedom))
        return bias


    def bias_estimate(self, r2tu_w , r2yu_tw, estimate):
        """
        Calculates the bias adjusted estimate
        :param r2tu_w: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yu_tw: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param estimate: estimate of regression

        :returns: the bias adjusted estimate
        """
        bias = self.bias(r2tu_w, r2yu_tw, self.standard_error, self.degree_of_freedom)
        if self.increase:
            return np.sign(estimate) * (abs(estimate) + bias)
        else:
            return np.sign(estimate) * (abs(estimate) - bias)


    def bias_se(self, r2tu_w, r2yu_tw ):
        """
        Calculates the bias adjusted standard error
        :param r2tu_w: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yu_tw: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out

        :returns:  the bias adjusted standard error
        """
        return np.sqrt((1 - r2yu_tw) / (1 - r2tu_w)) * self.standard_error * np.sqrt(self.degree_of_freedom / (self.degree_of_freedom - 1))


    def bias_t(self, r2tu_w, r2yu_tw, h0, estimate):
        """
        Calculates the bias adjusted t-value
        :param r2tu_w: partial R^2  of putative unobserved confounder "z" with treatment "d", observed covariates "x" partialed out
        :param r2yu_tw: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param h0: hypothesis value
        :param estimate: estimate of regression

        :returns:  the bias adjusted t-value
        """
        new_estimate = self.bias_estimate(r2tu_w = r2tu_w, r2yu_tw = r2yu_tw, estimate = estimate)
        new_se = self.bias_se(r2tu_w = r2tu_w, r2yu_tw = r2yu_tw)
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
        
        for i in range(len(self.r2tu_w)):
            plt.scatter(self.r2tu_w[i], self.r2yu_tw[i], color = 'red', marker = 'D', edgecolors = 'black')
            plt.annotate(label_names[i], (self.r2tu_w[i] + x_limit / 40.0 , self.r2yu_tw[i] + y_limit / 40.0))
            plt.annotate("({:1.3f})".format(bound_value[i]), (self.r2tu_w[i] + x_limit / 40.0, self.r2yu_tw[i] - y_limit / 40.0))

    def generate_label(self, benchmark_covariate):
        """
        Generate label names consisting of covariate name and appending multipliers frac_strength_outcome and frac_strength_treatment
        """
        label_names = []
        for i in range(len(self.frac_strength_treatment)):
            frac_strength_treatment = self.frac_strength_treatment[i]
            frac_strength_outcome = self.frac_strength_outcome[i]
            if frac_strength_treatment == frac_strength_outcome:
                mult_text = str(round(frac_strength_outcome,2))
            else:
                mult_text = str(round(frac_strength_treatment,2)) + '/' + str(round(frac_strength_outcome,2))
            label = mult_text + 'x' + ' ' + str(benchmark_covariate)
            label_names.append(label)
        return label_names


    def contour_plot(self,sensitivity_variable, frac_strength_treatment = 1,  estimate_threshold = 0, t_threshold = 2, contour_color = "black", threshold_color = "red", x_limit = None, y_limit = None):
        """
        The horizontal axis shows hypothetical values of the patial R2 of unobserved confounder(s) with the treatment
        The vertical axis shows hypothetical values of the patial R2 of unobserved confounder(s) with the outcome.
        The contour levels represent adjusted t-values or estimates of unobserved confounding.

        :param sensitivity_variable: "estimate" or "t-value"
        :param frac_strength_treatment: strength of association between benchmark and treatment variable to test with benchmark bounding
        :param r2yu_tw: partial R^2  of putative unobserved confounder "z" with outcome "y", observed covariates "x" and treatment "d" partialed out
        :param estimate_threshold: threshold line to emphasize when contours correspond to estimate, (default = 0)
        :param t_threshold: threshold line to emphasize when contours correspond to t-value, (default = 2)
        :param x_limit: x axis maximum
        :param y_limit: y axis maximum
        :param contour_color: color of contour line(default = black)
        :param threshold_color: color of threshold line(default = red)
        """

        if self.increase:
            estimate_threshold = self.estimate * (1 + self.percent_change_rvalue)
        else:
            estimate_threshold = self.estimate * (1 - self.percent_change_rvalue)
        
        t_threshold = abs(t.ppf(self.significance_level / 2, self.degree_of_freedom - 1)) * np.sign(self.stats['t_statistic'])
        if x_limit is None or y_limit is None:
            if self.r2tu_w is not None:
                x_limit = round(max(self.r2tu_w)*3,1) + 0.2
            if self.r2yu_tw is not None:
                y_limit = round(max(self.r2yu_tw)*3,1) + 0.2
            if(x_limit > y_limit):
                y_limit = x_limit
            else:
                x_limit = y_limit
        grid_x = np.arange(0.0, x_limit, x_limit/400)
        grid_y = np.arange(0.0, y_limit, y_limit/400)

        bound_value = None

        if sensitivity_variable == 'estimate':
            z_axis = [[self.bias_estimate(grid_x[j], grid_y[i], self.estimate) for j in range(len(grid_x))] for i in range(len(grid_y))]
            threshold = estimate_threshold
            plot_estimate = self.estimate
            bound_value = self.bias_estimate(self.r2tu_w, self.r2yu_tw, self.estimate)
        else:
            z_axis = [[self.bias_t(grid_x[j], grid_y[i], estimate_threshold, self.estimate) for j in range(len(grid_x))] for i in range(len(grid_y))]
            threshold = t_threshold
            plot_estimate = (self.estimate - estimate_threshold) / self.standard_error
            bound_value = self.bias_t(self.r2tu_w, self.r2yu_tw, estimate_threshold, self.estimate)
        
        #print(z_axis)
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

        label_names = self.generate_label(self.original_benchmark_covariates)
        
        if self.r2tu_w is not None:
            self.add_bound(bound_value, sensitivity_variable, label_names, x_limit, y_limit)
        
        margin_x = 0.05 * x_limit
        margin_y = 0.05 * y_limit
        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0, x1 + margin_x, y0, y1 + margin_y))
        plt.tight_layout()
    
    def __str__(self):
        s = "Sensitivity Analysis to Unobserved Confounding\n\n"
        s += "Unadjusted Estimates of {0} :\n".format(self.original_treatment_name)
        s += "Coefficient Estimate : {0}\n".format(self.estimate)
        s += "Standard Error : {0}\n".format(self.standard_error)
        s += "t-value : {0}\n\n".format(self.stats['t_statistic'])
        s += "Sensitivity Statistics : \n"
        s += "Partial R2 of treatment with outcome : {0}\n".format(self.stats['r2yd_x'])
        s += "Robustness Value : {0}\n\n".format(self.stats['robustness_value'])
        s += "Verbal Interpretation of results :\n"
        s += "Any confounder explaining less than {0}% percent of the residual variance of both the treatment and the outcome would not be strong enough to bring down the estimated effect to 0\n\n".format(round(self.stats['robustness_value'] * 100, 2))
        s += "For a significance level of {0}%, any confounder explaining more than {1}% percent of the residual variance of both the treatment and the outcome would be strong enough to make the estimated effect not 'statistically significant'\n\n".format(self.significance_level*100,round(self.stats['robustness_value_alpha'] * 100, 2))
        s += "If confounders explained 100% of the residual variance of the outcome, they would need to explain at least {0}% of the residual variance of the treatment to bring down the estimated effect to 0\n".format(round(self.stats['r2yd_x'] * 100, 2))
        return s
        
