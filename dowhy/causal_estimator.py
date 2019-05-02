import logging

import numpy as np
import sympy as sp


class CausalEstimator:
    """Base class for an estimator of causal effect.

    """

    def __init__(self, data, identified_estimand, treatment, outcome,
                 test_significance, params=None):
        """Initializes an estimator with data and names of relevant variables.

        More description.

        :param data: data frame containing the data
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param params: (optional) additional method parameters
        :returns: an instance of the estimator class.

        """
        self._data = data
        self._target_estimand = identified_estimand
        # Currently estimation methods only support univariate treatment and outcome
        self._treatment_name = treatment[0]
        self._outcome_name = outcome[0]
        self._significance_test = test_significance
        self._estimate = None
        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

        self.logger = logging.getLogger(__name__)

    def estimate_effect(self):
        """TODO.

        More description.

        :param self: object instance of class Estimator
        :returns: point estimate of causal effect

        """
        self._treatment = self._data[self._treatment_name]
        self._outcome = self._data[self._outcome_name]
        est = self._estimate_effect()
        # self._estimate = est


        if self._significance_test:
            signif_dict = self.test_significance(est)
            est.add_significance_test_results(signif_dict)
        return est

    def _do(self, x):
        raise NotImplementedError

    def do(self, x):
        """TODO.

        More description.

        :param arg1:
        :returns:

        """
        self._treatment = self._data[self._treatment_name]
        self._outcome = self._data[self._outcome_name]
        est = self._do(x)

        return est

    def test_significance(self, estimate, num_simulations=1000):
        """Test statistical significance of obtained estimate.

        Uses resampling to create a non-parametric significance test.
        A general procedure. Individual estimators can override this method.

        :param self: object instance of class Estimator
        :param estimate: obtained estimate
        :returns:

        """
        if not hasattr(self,'num_simulations'):
            self.num_simulations = num_simulations
        null_estimates = np.zeros(self.num_simulations)
        for i in range(self.num_simulations):
            self._outcome = np.random.permutation(self._outcome)
            est = self._estimate_effect()
            null_estimates[i] = est.value

        sorted_null_estimates = np.sort(null_estimates)
        self.logger.debug("Null estimates: {0}".format(sorted_null_estimates))
        median_estimate = sorted_null_estimates[int(self.num_simulations / 2)]
        # Doing a two-sided test
        if estimate.value > median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate.value, side="left")
            p_value = 1 - (estimate_index / self.num_simulations)
        if estimate.value <= median_estimate:
            # Being conservative with the p-value reported
            estimate_index = np.searchsorted(sorted_null_estimates, estimate.value, side="right")
            p_value = (estimate_index / self.num_simulations)
        signif_dict = {
            'p_value': p_value,
            'sorted_null_estimates': sorted_null_estimates
        }
        return signif_dict


class CausalEstimate:
    """TODO.

    """

    def __init__(self, estimate, target_estimand, realized_estimand_expr, **kwargs):
        self.value = estimate
        self.target_estimand = target_estimand
        self.realized_estimand_expr = realized_estimand_expr
        self.params = kwargs
        self.significance_test = None

    def add_significance_test_results(self, test_results):
        self.significance_test = test_results

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

