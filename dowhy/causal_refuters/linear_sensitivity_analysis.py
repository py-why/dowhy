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
    :param h0: hypothesis value
    :param increase: True implies that confounder increases the absolute value of estimate and vice versa. (Default = False)
    :param benchmark_covariates: names of variables for bounding strength of confounders
    :param significance_level: confidence interval for statistical inference(default = 0.05)
    :param frac_strength_treatment: strength of association between unobserved confounder and treatment compared to benchmark covariate
    :param frac_strength_outcome: strength of association between unobserved confounder and outcome compared to benchmark covariate
    :param common_causes_order: The order of column names in OLS regression data
    """

    def __init__(self, OLSmodel = None, data = None, treatment_name = None, percent_change_rvalue = 1.0, significance_level = 0.05, increase = False, benchmark_covariates = None, h0 = 0, frac_strength_treatment = None, frac_strength_outcome = None, common_causes_order = None):
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
        self.h0 = h0

        # common_causes_map : maps the original variable names to variable names in OLS regression
        self.common_causes_map = {}
        for i in range(len(common_causes_order)):
            self.common_causes_map[common_causes_order[i]] = "x"+str(len(self.treatment_name)+i+1)

        # benchmark_covariates: stores variable names in terms of regression model variables
        benchmark_covariates = parse_state(benchmark_covariates)
        self.benchmark_covariates = []
        # original_benchmark_covariates: stores original variable names for labelling
        self.original_benchmark_covariates = benchmark_covariates
        for i in range(len(benchmark_covariates)):
            self.benchmark_covariates.append(self.common_causes_map[benchmark_covariates[i]])

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
        # r2tu_w: partial R^2  of unobserved confounder "u" with treatment "t", after controlling for observed covariates "w"
        self.r2tu_w = None
        # r2yu_tw: partial R^2  of unobserved confounder "u" with outcome "y", after controlling for observed covariates "w" and treatment "t"
        self.r2yu_tw = None
        # r2twj_w: partial R^2 of observed covariate wj with treatment "t", after controlling for observed covariates "w" excluding wj
        self.r2twj_w = None
        # r2ywj_tw:  partial R^2 of observed covariate wj with outcome "y", after controlling for observed covariates "w" (excluding wj) and treatment "t"
        self.r2ywj_tw = None
        # bias adjusted terms
        self.bias_adjusted_estimate = None
        self.bias_adjusted_se = None
        self.bias_adjusted_t = None
        self.bias_adjusted_lower_CI = None
        self.bias_adjusted_upper_CI = None
        # benchmarking_results: dataframe containing information about bounds and bias adjusted terms
        self.benchmarking_results = None
        # stats: dictionary containing information like robustness value, partial R^2, estimate, standard error , degree of freedom, partial f^2, t-statistic
        self.stats = None
        self.logger = logging.getLogger(__name__)

    
    def treatment_regression(self, model = None, treatment = None):
        """
        Function to perform regression with treatment as outcome

        :param model: original OLS regression model
        :param treatment: treatment variable names

        :returns: new OLS regression model
        """
        data = pd.DataFrame(model.model.exog, columns = model.model.exog_names)
        treatment_data = np.array(data[treatment])
        non_treatment_data = data.drop(columns = treatment)
        non_treatment_data.insert(0,0,1) #inserting bias
        new_model = sma.OLS(treatment_data, non_treatment_data)
        OLSmodel_new = new_model.fit()
        
        return OLSmodel_new

    
    def partial_r2_func(self, model = None, treatment = None):
        """
        Computes the partial R^2 of regression model 

        :param model: statsmodels OLS regression model
        :param treatment: treatment name

        :returns: partial R^2 value
        """

        estimate = model.params[treatment]
        degree_of_freedom = int(model.df_resid)

        if np.isscalar(estimate): #for single covariate
            t_stats = model.tvalues[treatment]
            return  (t_stats ** 2 / (t_stats ** 2 + degree_of_freedom))

        else: #compute for a group of covariates
            covariance_matrix = model.cov_params().loc[treatment, :][treatment]
            n = len(estimate) #number of parameters in model
            f_stat = np.matmul(np.matmul(estimate.values.T, np.linalg.inv(covariance_matrix.values)), estimate.values) / n
            return f_stat * n / (f_stat * n + degree_of_freedom)

            
    def robustness_value_func(self, model = None, alpha = 1.0):
        """
        Function to calculate the robustness value. 
        It is the minimum strength of association that confounders must have with treatment and outcome to change conclusions.
        RVq describes how strong the association must be in order to reduce the estimated effect by (100 * percent_change_rvalue)%.
        RVq close to 1 means the treatment effect can handle strong confounders explaining  almost all residual variation of the treatment and the outcome.
        RVq close to 0 means that even very weak confounders can also change the results.

        :param model: OLS regression model
        :param alpha: confidence interval (default = 1)

        :returns: robustness value 
        """

        partial_cohen_f = abs(self.t_stats / np.sqrt(self.degree_of_freedom)) #partial f of treatment t with outcome y. f = t_val/sqrt(dof)
        f_q = self.percent_change_rvalue * partial_cohen_f
        t_alpha_df_1 = t.ppf(alpha / 2 , self.degree_of_freedom - 1) # t-value threshold with alpha significance level and dof-1 degrees of freedom
        f_critical = abs(t_alpha_df_1) / np.sqrt(self.degree_of_freedom - 1)
        f_adjusted = f_q - f_critical

        if f_adjusted < 0:
            r_value = 0
        else:
            r_value = 0.5 * (np.sqrt(f_adjusted ** 4 + (4 * f_adjusted ** 2)) - f_adjusted ** 2)

        if f_adjusted > 0 and f_q > 1 / f_critical:
            r_value = (f_q ** 2 - f_critical ** 2) / (1 + f_q ** 2)

        return r_value


    def compute_bias_adjusted(self, r2tu_w, r2yu_tw):
        """
        Computes the bias adjusted estimate, standard error, t-value,  partial R2, confidence intervals

        :returns: Python dictionary with information about partial R^2 of confounders with treatment and outcome and bias adjusted variables
        """

        bias_factor = np.sqrt((r2yu_tw * r2tu_w) / (1 - r2tu_w))
        bias = bias_factor * (self.standard_error * np.sqrt(self.degree_of_freedom))

        if self.increase:
            bias_adjusted_estimate = np.sign(self.estimate) * (abs(self.estimate) + bias)
        else:
            bias_adjusted_estimate = np.sign(self.estimate) * (abs(self.estimate) - bias)

        bias_adjusted_se = np.sqrt((1 - r2yu_tw) / (1 - r2tu_w)) * self.standard_error * np.sqrt(self.degree_of_freedom / (self.degree_of_freedom - 1))

        bias_adjusted_t = (bias_adjusted_estimate - self.h0) / bias_adjusted_se

        bias_adjusted_partial_r2 = bias_adjusted_t ** 2 / (bias_adjusted_t ** 2 + (self.degree_of_freedom - 1)) #partial r2 formula used with new t value and dof - 1

        num_se = t.ppf(self.significance_level / 2, self.degree_of_freedom) # Number of standard errors within Confidence Interval

        bias_adjusted_upper_CI = bias_adjusted_estimate - num_se * bias_adjusted_se
        bias_adjusted_lower_CI = bias_adjusted_estimate + num_se * bias_adjusted_se

        benchmarking_results ={
                'r2tu_w': r2tu_w,
                'r2yu_tw': r2yu_tw,
                'bias_adjusted_estimate': bias_adjusted_estimate,
                'bias_adjusted_se': bias_adjusted_se,
                'bias_adjusted_t': bias_adjusted_t,
                'bias_adjusted_lower_CI': bias_adjusted_lower_CI,
                'bias_adjusted_upper_CI': bias_adjusted_upper_CI
            }

        return benchmarking_results
    
    def perform_analysis(self):
        """
        Function to perform sensitivity analysis. 
        By default it generates a plot of point estimate and the variations with respect to benchmarking and unobserved confounding.
        """

        self.standard_error = list(self.OLSmodel.bse[1:(len(self.treatment_name)+1)])[0]
        self.degree_of_freedom = int(self.OLSmodel.df_resid)
        self.estimate = list(self.OLSmodel.params[1:(len(self.treatment_name)+1)])[0]
        self.t_stats = list(self.OLSmodel.tvalues[self.treatment_name])[0]

        # partial R^2 (r2yt_w) is the proportion of variation in outcome uniquely explained by treatment
        partial_r2 = self.partial_r2_func(self.OLSmodel, self.treatment_name)
        RVq = self.robustness_value_func(model = self.OLSmodel)
        RVqa = self.robustness_value_func(model = self.OLSmodel, alpha = self.significance_level)

        if self.increase:  
            self.h0 = self.estimate * (1 + self.percent_change_rvalue) 
        else:
            self.h0 = self.estimate * (1 - self.percent_change_rvalue)

        self.t_stats = (self.estimate - self.h0 ) / self.standard_error
        self.partial_f2 = self.t_stats ** 2 / self.degree_of_freedom
        
        # build a new regression model by considering treatment variables as outcome 
        OLS_model_new = self.treatment_regression(model = self.OLSmodel, treatment = self.treatment_name)

        # r2twj_w is partial R^2 of covariate wj with treatment "t", after controlling for covariates w(excluding wj)
        # r2ywj_tw is partial R^2 of covariate wj with outcome "y", after controlling for covariates w(excluding wj) and treatment "t"
        self.r2twj_w = []
        self.r2ywj_tw = []
        for covariate in self.benchmark_covariates:
            self.r2ywj_tw.append(self.partial_r2_func(self.OLSmodel, covariate)) 
            self.r2twj_w.append(self.partial_r2_func(OLS_model_new, covariate))
        
        for i in range(len(self.benchmark_covariates)):
            r2twj_w = self.r2twj_w[i]
            r2ywj_tw = self.r2ywj_tw[i]
            
            # r2tu_w is the partial r^2 from regressing u on t after controlling for w
            self.r2tu_w = self.frac_strength_treatment * (r2twj_w / (1 - r2twj_w))
            if(any(val >= 1 for val in self.r2tu_w)):
                raise ValueError("r2tu_w can not be >= 1. Try a lower frac_strength_treatment value")

            r2uwj_wt = self.frac_strength_treatment * (r2twj_w ** 2) / ((1 - self.frac_strength_treatment * r2twj_w) * (1 - r2twj_w))
            if(any(val >= 1 for val in r2uwj_wt)):
                raise ValueError("r2uwj_wt can not be >= 1. Try a lower frac_strength_treatment value")

            self.r2yu_tw = ((np.sqrt(self.frac_strength_outcome) + np.sqrt(r2uwj_wt)) / np.sqrt(1 - r2uwj_wt)) ** 2 * (r2ywj_tw / (1 - r2ywj_tw))
            if(any(val > 1 for val in self.r2yu_tw)):
                for i in range(len(self.r2yu_tw)):
                    if self.r2yu_tw[i]>1:
                        self.r2yu_tw[i]=1
                self.logger.warning("Warning: r2yu_tw can not be > 1. Try a lower frac_strength_treatment. Setting r2yu_tw to 1")

            #Compute bias adjusted terms
        
        self.benchmarking_results = self.compute_bias_adjusted(self.r2tu_w, self.r2yu_tw)
        self.bias_adjusted_estimate = self.benchmarking_results['bias_adjusted_estimate']
        self.bias_adjusted_se = self.benchmarking_results['bias_adjusted_se']
        self.bias_adjusted_t = self.benchmarking_results['bias_adjusted_t']
        self.bias_adjusted_lower_CI = self.benchmarking_results['bias_adjusted_lower_CI']
        self.bias_adjusted_upper_CI = self.benchmarking_results['bias_adjusted_upper_CI']
        
        self.plot()

        self.stats = {
            'estimate' : self.estimate,
            'standard_error' : self.standard_error,
            'degree of freedom': self.degree_of_freedom,
            't_statistic' : self.t_stats,
            'r2yt_w' : partial_r2,
            'partial_f2' : self.partial_f2,
            'robustness_value' : RVq,
            'robustness_value_alpha' : RVqa 
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

        critical_estimate = self.h0
        contour_values = np.zeros((len(r2yu_tw), len(r2tu_w)))
        for i in range(len(r2yu_tw)):
            y = r2tu_w[i]
            for j in range(len(r2tu_w)):
                x = r2yu_tw[j]
                benchmarking_results = self.compute_bias_adjusted(r2tu_w = x, r2yu_tw = y)
                estimate = benchmarking_results['bias_adjusted_estimate']
                contour_values[i][j] = estimate

        estimate_bounds = self.bias_adjusted_estimate
        return contour_values, critical_estimate, estimate_bounds

    
    def plot_t(self,r2tu_w, r2yu_tw):
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

        t_alpha_df_1 = t.ppf(self.significance_level / 2 , self.degree_of_freedom - 1) # t-value threshold with alpha significance level and dof-1 degrees of freedom
        critical_t = abs(t_alpha_df_1) * np.sign(self.t_stats)

        contour_values = []
        for x in r2tu_w:
            contour = []
            for y in r2yu_tw:
                benchmarking_results = self.compute_bias_adjusted(r2tu_w = x, r2yu_tw = y)
                t_value = benchmarking_results['bias_adjusted_t']
                contour.append(t_value)
            contour_values.append(contour)

        t_bounds = self.bias_adjusted_t
        return contour_values, critical_t, t_bounds

    
    def plot(self, sensitivity_variable = 'estimate', critical_estimate = 0, critical_t = 2, contour_color = "blue", threshold_color = "red", x_limit = 0.8, y_limit = 0.8 ):
        """
        The horizontal axis shows hypothetical values of the partial R2 of unobserved confounder(s) with the treatment
        The vertical axis shows hypothetical values of the partial R2 of unobserved confounder(s) with the outcome.
        The contour levels represent adjusted t-values or estimates for unobserved confounders with these hypothetical partialR2 values.
        We also plot bounds on the partial R^2 of the unobserved confounders obtained from observed covariates

        :param sensitivity_variable: "estimate" or "t-value"
        :param critical_estimate: threshold line to represent the contour corresponding to estimate, (default = 0)
        :param critical_t: threshold line to represent the contour corresponding to t-value, (default = 2 - usual approx value for 95% CI)
        :param x_limit: plot's maximum x_axis value
        :param y_limit: plot's minimum y_axis value
        :param contour_color: color of contour line(default = blue)
        :param threshold_color: color of threshold line(default = red)
        """

        #Plotting the contour plot
        fig, ax = plt.subplots(1, 1, figsize = (7,7))
        ax.set_title("Sensitivity contour plot of %s"  %sensitivity_variable)
        ax.set_xlabel("Partial R^2 of confounder with treatment")
        ax.set_ylabel("Partial R^2 of confounder with outcome")

        for i in range(len(self.r2tu_w)):
            x = self.r2tu_w[i]
            y = self.r2yu_tw[i]
            if(x > 0.8 or y > 0.8):
                x_limit = 0.95
                y_limit = 0.95
                break

        r2tu_w = np.arange(0.0, x_limit, x_limit / 200)
        r2yu_tw = np.arange(0.0, y_limit, y_limit / 200)

        unadjusted_point_estimate = None

        if sensitivity_variable == "estimate":
            contour_values, contour_threshold, bound_values  = self.plot_estimate(r2tu_w, r2yu_tw)
            unadjusted_estimate = self.estimate
            unadjusted_point_estimate = unadjusted_estimate
        else:
            contour_values, contour_threshold, bound_values = self.plot_t(r2tu_w, r2yu_tw)
            unadjusted_t = self.t_stats
            unadjusted_point_estimate = unadjusted_t
        
        #Adding contours
        contour_plot = ax.contour(r2tu_w, r2yu_tw, contour_values, colors = contour_color, linewidths = 0.75, linestyles = "solid")
        ax.clabel(contour_plot, inline = 1, fontsize = 9, colors = "black")
        
        #Adding threshold contour line
        contour_plot = ax.contour(r2tu_w, r2yu_tw, contour_values, colors = threshold_color, linewidths = 0.75, levels = [contour_threshold])

        #Adding unadjusted point estimate 
        ax.scatter([0],[0], marker = 'D', color = "black", label = "Unadjusted({:1.2f})".format(unadjusted_point_estimate))

        #Adding bounds to partial R^2 values for given strength of confounders
        for i in range(len(self.frac_strength_treatment)):
            frac_strength_treatment = self.frac_strength_treatment[i]
            frac_strength_outcome = self.frac_strength_outcome[i]
            if(frac_strength_treatment == frac_strength_outcome):
                signs = str(round(frac_strength_treatment,2))
            else:
                signs = str(round(frac_strength_treatment,2)) + '/' + str(round(frac_strength_outcome,2))
            label = str(i+1) + "  "+signs + ' X ' + str(self.original_benchmark_covariates) + " ({:1.2f}) ".format(bound_values[i])
            ax.scatter(self.r2tu_w[i], self.r2yu_tw[i], color = 'red', marker = '^', label = label)
            ax.annotate(str(i+1), (self.r2tu_w[i] + 0.005, self.r2yu_tw[i] + 0.005 ))

        ax.legend(bbox_to_anchor = (1.6 , 0.6))
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
        s += "Partial R2 of treatment with outcome : {0}\n".format(self.stats['r2yt_w'])
        s += "Robustness Value : {0}\n\n".format(self.stats['robustness_value'])
        s += "Interpretation of results :\n"
        s += "Any confounder explaining less than {0}% percent of the residual variance of both the treatment and the outcome would not be strong enough to explain away the observed effect i.e bring down the estimate to 0 \n\n".format(round(self.stats['robustness_value'] * 100, 2))
        s += "For a significance level of {0}%, any confounder explaining more than {1}% percent of the residual variance of both the treatment and the outcome would be strong enough to make the estimated effect not 'statistically significant'\n\n".format(self.significance_level*100,round(self.stats['robustness_value_alpha'] * 100, 2))
        s += "If confounders explained 100% of the residual variance of the outcome, they would need to explain at least {0}% of the residual variance of the treatment to bring down the estimated effect to 0\n".format(round(self.stats['r2yt_w'] * 100, 2))
        return s


        

        






