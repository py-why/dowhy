import logging

import numpy as np
import pandas as pd
import sympy as sp

from sklearn.utils import resample

class CausalEstimator:
    """Base class for an estimator of causal effect.

    Subclasses implement different estimation methods. All estimation methods are in the package "dowhy.causal_estimators"

    """
    # The default number of simulations for statistical testing
    DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST = 1000
    # The default number of simulations to obtain confidence intervals
    DEFAULT_NUMBER_OF_SIMULATIONS_CI = 100
    # The portion of the total size that should be taken each time to find the confidence interval
    # 1 is the recommended value 
    # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    # https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214 
    DEFAULT_SAMPLE_SIZE_FRACTION = 1

    def __init__(self, data, identified_estimand, treatment, outcome,
                 control_value=0, treatment_value=1,
                 test_significance=False, evaluate_effect_strength=False,
                 confidence_intervals = False,
                 target_units=None, effect_modifiers=None,
                 params=None):
        """Initializes an estimator with data and names of relevant variables.

        More description.

        :param data: data frame containing the data
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param control_value: Value of the treatment in the control group, for effect estimation.  If treatment is multi-variate, this can be a list.
        :param treatment_value: Value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list.
        :param test_significance: whether to test significance
        :param evaluate_effect_strength: (Experimental) whether to evaluate the strength of effect
        :param confidence_intervals: (Experimental) Binary flag indicating whether confidence intervals should be computed.
        :param target_units: (Experimental) The units for which the treatment effect should be estimated. This can be a string for common specifications of target units (namely, "ate", "att" and "atc"). It can also be a lambda function that can be used as an index for the data (pandas DataFrame). Alternatively, it can be a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
        :param effect_modifiers: variables on which to compute separate effects, or return a heterogeneous effect function. Not all methods support this currently.
        :param params: (optional) additional method parameters
            num_simulations: The number of simulations for testing the statistical significance of the estimator
            num_ci_simulations: The number os simulations for finding the confidence estimate for a estimate
            sample_size_fraction: The size of the sample for the bootstrap estimator
        :returns: an instance of the estimator class.

        """
        self._data = data
        self._target_estimand = identified_estimand
        # Currently estimation methods only support univariate treatment and outcome
        self._treatment_name = treatment
        self._outcome_name = outcome[0] # assuming one-dimensional outcome
        self._control_value = control_value
        self._treatment_value = treatment_value
        self._significance_test = test_significance
        self._effect_strength_eval = evaluate_effect_strength
        self._target_units = target_units
        self._effect_modifier_names = effect_modifiers
        self._confidence_intervals = confidence_intervals
        self._estimate = None
        self._effect_modifiers = None
        self.method_params = params

        # Unpacking the keyword arguments
        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

        
        self.logger = logging.getLogger(__name__)

        # Checking if some parameters were set, otherwise setting to default values
        if not hasattr(self, 'num_simulations'):
            self.num_simulations = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST

        if not hasattr(self, 'num_ci_simulations') and self._confidence_intervals:
            self.num_ci_simulations = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI

        if not hasattr(self, 'sample_size_fraction') and self._confidence_intervals:
            self.sample_size_fraction = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION

        # Setting more values
        if self._data is not None:
            self._treatment = self._data[self._treatment_name]
            self._outcome = self._data[self._outcome_name]

        # Now saving the effect modifiers
        if self._effect_modifier_names:
            self._effect_modifiers = self._data[self._effect_modifier_names]
            self._effect_modifiers = pd.get_dummies(self._effect_modifiers, drop_first=True)
            self.logger.debug("Effect modifiers: " +
                          ",".join(self._effect_modifier_names))
    
    @staticmethod
    def get_estimator_object(new_data, identified_estimand, estimate):
        '''
            This method is used to create a new estimator, of the same type as the
            one passed in the estimate argument. It then attempts to create a new object
            with the new_data and the identified_estimand

            Parameters
            -----------

            'new_data': np.ndarray, pd.Series, pd.DataFrame 
            The newly assigned data on which the estimator should run

            'identified_estimand': IdentifiedEstimand
            An instance of the identified estimand class that provides the information with 
            respect to which causal pathways are employed when the treatment effects the outcome
            'estimate': CausalEstimate
            It is an already existing estimator whose properties we wish to replicate
        '''
        estimator_class = estimate.params['estimator_class']
        new_estimator = estimator_class(
                new_data,
                identified_estimand,
                identified_estimand.treatment_variable, identified_estimand.outcome_variable, #names of treatment and outcome
                test_significance=None,
                evaluate_effect_strength=False,
                confidence_intervals = estimate.params["confidence_intervals"],
                target_units = estimate.params["target_units"],
                effect_modifiers = estimate.params["effect_modifiers"],
                params = estimate.params["method_params"]
                )
        return new_estimator

    
    def _estimate_effect(self):
        '''
            This method is to be overriden by the child classes, so that they can run the estimation technique of their choice
        '''
        raise NotImplementedError

    def estimate_effect(self):
        """Base estimation method that calls the estimate_effect method of its calling subclass.

        Can optionally also test significance and estimate effect strength for any returned estimate.

        TODO: Enable methods to return a confidence interval in addition to the point estimate.

        :param self: object instance of class Estimator
        :returns: point estimate of causal effect

        """

        est = self._estimate_effect()
        self._estimate = est

        if self._significance_test:
            signif_dict = self.test_significance(est)
            est.add_significance_test_results(signif_dict)
        if self._confidence_intervals:
            confidence_intervals = self.get_confidence_intervals(est)
            est.add_confidence_intervals(confidence_intervals)
        if self._effect_strength_eval:
            effect_strength_dict = self.evaluate_effect_strength(est)
            est.add_effect_strength(effect_strength_dict)

        return est

    def estimate_effect_naive(self):
        #TODO Only works for binary treatment
        df_withtreatment = self._data.loc[self._data[self._treatment_name] == 1]
        df_notreatment = self._data.loc[self._data[self._treatment_name]== 0]
        est = np.mean(df_withtreatment[self._outcome_name]) - np.mean(df_notreatment[self._outcome_name])
        return CausalEstimate(est, None, None)

    def _do(self, x):
        raise NotImplementedError

    def do(self, x):
        """Method that implements the do-operator.

        Given a value x for the treatment, returns the expected value of the outcome when the treatment is intervened to a value x.

        :param x: Value of the treatment
        :returns: Value of the outcome when treatment is intervened/set to x.

        """
        est = self._do(x)
        return est

    def construct_symbolic_estimator(self, estimand):
        raise NotImplementedError

    def get_confidence_intervals(self, estimate):
        '''
            Find the confidence intervals corresponding to any estimator
            This is done with the help of bootstrapped confidence intervals

            For more details, refer to the following links:
            https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
            https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214
        '''

        # The array that stores the results of all estimations
        simulation_results = np.zeros(self.num_ci_simulations)

        # Find the sample size the proportion with the population size
        sample_size= int( self.sample_size_fraction * len(self._data) )

        if sample_size > len(self._data):
            self.logger.warning("WARN: The sample size is greater than the data being sampled")
        
        self.logger.info("INFO: The sample size: {}".format(sample_size) )
        self.logger.info("INFO: The number of simulations: {}".format(self.num_ci_simulations) )
        
        # Perform the set number of simulations
        for index in range(self.num_ci_simulations):
            new_data = resample(self._data,n_samples=sample_size)

            new_estimator = CausalEstimator.get_estimator_object(new_data, self._target_estimand, self._estimate)
            new_effect = new_estimator.estimate_effect()
            simulation_results[index] = new_effect.value
        
        # Now use the data obtained from the simulations to get the value of the confidence estimates
        # Sort the simulations
        simulation_results.sort()
        # Now we take the 5th and the 95th values
        lower_bound_index = int( 0.05 * len(simulation_results) ) 
        upper_bound_index = int( 0.95 * len(simulation_results) )
        
        # get the values
        lower_bound = simulation_results[lower_bound_index]
        upper_bound = simulation_results[upper_bound_index]

        return (lower_bound, upper_bound)

    def test_significance(self, estimate, num_simulations=None):
        """Test statistical significance of obtained estimate.

        Uses resampling to create a non-parametric significance test.
        A general procedure. Individual estimators can override this method.

        :param self: object instance of class Estimator
        :param estimate: obtained estimate
        :param num_simulations: (optional) number of simulations to run
        :returns:

        """
        if num_simulations is None:
            num_simulations = self.num_simulations
        null_estimates = np.zeros(num_simulations)
        for i in range(num_simulations):
            self._outcome = np.random.permutation(self._outcome)
            est = self._estimate_effect()
            null_estimates[i] = est.value

        sorted_null_estimates = np.sort(null_estimates)
        self.logger.debug("Null estimates: {0}".format(sorted_null_estimates))
        median_estimate = sorted_null_estimates[int(num_simulations / 2)]
        # Doing a two-sided test
        if estimate.value > median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate.value, side="left")
            p_value = 1 - (estimate_index / num_simulations)
        if estimate.value <= median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate.value, side="right")
            p_value = (estimate_index / num_simulations)
        signif_dict = {
            'p_value': p_value,
            'sorted_null_estimates': sorted_null_estimates
        }
        return signif_dict

    def evaluate_effect_strength(self, estimate):
        fraction_effect_explained = self._evaluate_effect_strength(estimate, method="fraction-effect")
        # Need to test r-squared before supporting
        #effect_r_squared = self._evaluate_effect_strength(estimate, method="r-squared")
        strength_dict = {
                'fraction-effect': fraction_effect_explained
         #       'r-squared': effect_r_squared
                }
        return strength_dict

    def _evaluate_effect_strength(self, estimate, method="fraction-effect"):
        supported_methods = ["fraction-effect"]
        if method not in supported_methods:
            raise NotImplementedError("This method is not supported for evaluating effect strength")
        if method == "fraction-effect":
            naive_obs_estimate = self.estimate_effect_naive()
            print(estimate.value, naive_obs_estimate.value)
            fraction_effect_explained = estimate.value/naive_obs_estimate.value
            return fraction_effect_explained
        #elif method == "r-squared":
        #    outcome_mean = np.mean(self._outcome)
        #    total_variance = np.sum(np.square(self._outcome - outcome_mean))
            # Assuming a linear model with one variable: the treatment
            # Currently only works for continuous y
        #    causal_model = outcome_mean + estimate.value*self._treatment
        #    squared_residual = np.sum(np.square(self._outcome - causal_model))
        #    r_squared = 1 - (squared_residual/total_variance)
        #    return r_squared
        else:
            return None

class CausalEstimate:
    """Class for the estimate object that every causal estimator returns

    """

    def __init__(self, estimate, target_estimand, realized_estimand_expr, **kwargs):
        self.value = estimate
        self.target_estimand = target_estimand
        self.realized_estimand_expr = realized_estimand_expr
        self.params = kwargs
        if self.params is not None:
            for key, value in self.params.items():
                setattr(self, key, value)

        self.significance_test = None
        self.effect_strength = None

    def add_significance_test_results(self, test_results):
        self.significance_test = test_results

    def add_confidence_intervals(self, confidence_intervals):
        self.confidence_intervals = confidence_intervals

    def add_effect_strength(self, strength_dict):
        self.effect_strength = strength_dict

    def add_params(self, **kwargs):
        self.params.update(kwargs)

    def __str__(self):
        s = "*** Causal Estimate ***\n"
        s += "\n## Target estimand\n{0}".format(self.target_estimand)
        s += "\n## Realized estimand\n{0}".format(self.realized_estimand_expr)
        s += "\n## Estimate\n"
        s += "Value: {0}\n".format(self.value)
        if self.significance_test is not None:
            s += "\n## Statistical Significance\n"
            if self.significance_test["p_value"]>0:
                s += "p-value: {0}\n".format(self.significance_test["p_value"])
            else:
                s+= "p-value: <{0}\n".format(1/len(self.significance_test["sorted_null_estimates"]))
        if self.effect_strength is not None:
            s += "\n## Effect Strength\n"
            s += "Change in outcome attributable to treatment: {}\n".format(self.effect_strength["fraction-effect"])
            #s += "Variance in outcome explained by treatment: {}\n".format(self.effect_strength["r-squared"])
        return s


class RealizedEstimand(object):

    def __init__(self, identified_estimand, estimator_name):
        self.treatment_variable = identified_estimand.treatment_variable
        self.outcome_variable = identified_estimand.outcome_variable
        self.backdoor_variables = identified_estimand.backdoor_variables
        self.instrumental_variables = identified_estimand.instrumental_variables
        self.estimand_type = identified_estimand.estimand_type
        self.estimand_expression = None
        self.assumptions = None
        self.estimator_name = estimator_name

    def update_assumptions(self, estimator_assumptions):
        self.assumptions = estimator_assumptions

    def update_estimand_expression(self, estimand_expression):
        self.estimand_expression = estimand_expression

    def __str__(self):
        s = "Realized estimand: {0}\n".format(self.estimator_name)
        s += "Realized estimand type: {0}\n".format(self.estimand_type)
        s += "Estimand expression:\n{0}\n".format(sp.pretty(self.estimand_expression))
        j = 1
        for ass_name, ass_str in self.assumptions.items():
            s += "Estimand assumption {0}, {1}: {2}\n".format(j, ass_name, ass_str)
            j += 1
        return s

