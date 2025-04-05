import logging

import numpy as np

from dowhy.causal_refuter import CausalRefutation, CausalRefuter
from dowhy.utils.cit import conditional_MI, partial_corr


class GraphRefuter(CausalRefuter):
    """
    Class for performing refutations on graph and storing the results
    """

    def __init__(
        self, data, method_name_discrete="conditional_mutual_information", method_name_continuous="partial_correlation"
    ):
        """
        Initialize data for graph refutation

        :param data:input dataset
        :param method_name_discrete: name of method for testing conditional independence in discrete data
        :param method_name_continuous: name of method for testing conditional independece in continuous data
        :returns : instance of GraphRefutation class
        """
        self._refutation_passed = None
        self._data = data
        self._method_name_discrete = method_name_discrete
        self._method_name_continuous = method_name_continuous
        self._false_implications = []  # List containing the implications from the graph which hold false for dataset
        self._true_implications = []  # List containing the implications from the graph which hold true for dataset
        self._results = {}  # A dictionary with key as test set and value as [p-value, test_result]
        self.logger = logging.getLogger(__name__)

    def set_refutation_result(self, number_of_constraints_model):
        """
        Method to set the result for graph refutation. Set true if there are no false implications else false
        """
        if (len(self._true_implications)) == number_of_constraints_model:
            self._refutation_passed = True
        elif len(self._false_implications) == 0:
            self._refutation_passed = True
            self.logger.warning("Some tests could not be run : config not supported")
        elif len(self._false_implications) > 0:
            self._refutation_passed = False

    def partial_correlation(self, x=None, y=None, z=None):
        stats = partial_corr(data=self._data, x=x, y=y, z=list(z))
        p_value = stats["p-val"]
        key = (x, y) + (z,)
        if p_value < 0.05:
            # Reject H0
            self._false_implications.append([x, y, z])
            self._results[key] = [p_value, False]
        else:
            self._true_implications.append([x, y, z])
            self._results[key] = [p_value, True]

    def conditional_mutual_information(self, x=None, y=None, z=None):
        cmi_val = conditional_MI(data=self._data, x=x, y=y, z=list(z))
        key = (x, y) + (z,)
        if cmi_val <= 0.05:
            self._true_implications.append([x, y, z])
            self._results[key] = [cmi_val, True]
        else:
            self._false_implications.append([x, y, z])
            self._results[key] = [cmi_val, False]

    def refute_model(self, independence_constraints):
        """
        Method to test conditional independence using the graph refutation object on the given testing set

        :param independence_constraints: List of implications to test the conditional independence on
        :returns : GraphRefutation object
        """

        refute = GraphRefutation(
            method_name_continuous=self._method_name_continuous, method_name_discrete=self._method_name_discrete
        )

        all_nodes = list(self._data.columns.values)
        discrete_columns = []
        continuous_columns = []
        binary_columns = []
        variable_type = dict()
        for node in all_nodes:
            if self._data[node].dtype == np.int64 or self._data[node].dtype == np.int32:
                discrete_columns.append(node)
                variable_type[node] = "discrete"
                if self._data[node].isin([0, 1]).all():
                    binary_columns.append(node)
                    variable_type[node] = "binary"
            else:
                continuous_columns.append(node)
                variable_type[node] = "continuous"
        for a, b, c in independence_constraints:
            if a in continuous_columns and b in continuous_columns and all(node in continuous_columns for node in c):
                # a, b and c are all continuous variables
                if self._method_name_continuous is None or self._method_name_continuous == "partial_correlation":
                    self.partial_correlation(x=a, y=b, z=c)
                else:
                    self.logger.error(
                        "Invalid conditional independence test for continuous data. Supported tests - partial_correlation"
                    )

            elif a in discrete_columns and b in discrete_columns and all(node in discrete_columns for node in c):
                # a, b and c are all discrete variables
                if self._method_name_discrete is None or self._method_name_discrete == "conditional_mutual_information":
                    self.conditional_mutual_information(x=a, y=b, z=c)
                else:
                    self.logger.error(
                        "Invalid conditional independence test for discrete data. Supported tests - conditional_mutual_information"
                    )

            elif (
                (a in continuous_columns or a in binary_columns)
                and (b in continuous_columns or b in binary_columns)
                and all(node in continuous_columns or node in binary_columns for node in c)
            ):
                # c is set of continuous and binary variables and
                #   1. either a and b is continuous and the other is binary
                #   2. both a and b are binary
                self.partial_correlation(x=a, y=b, z=c)

            elif all(node in discrete_columns for node in c) and (a in discrete_columns or b in discrete_columns):
                # c is discrete and
                # either a or b is continuous and the other is discrete
                self.conditional_mutual_information(x=a, y=b, z=c)

            elif a in discrete_columns and b in discrete_columns:
                # a and b are discrete and c is a mixture of discrete and continuous variables. We discretize c and calculate conditional mutual information
                self.conditional_mutual_information(x=a, y=b, z=c)

            else:
                key = (a, b) + (c,)
                self._results[key] = [None, "NotImplemented"]
                variable_types_c = []
                for var in c:
                    variable_types_c.append(variable_type[var])
                self.logger.warning(
                    "The following setting with {0} as {1}, {2} as {3}, {4} as {5} not supported".format(
                        a, variable_type[a], b, variable_type[b], c, variable_types_c
                    )
                )

        self.set_refutation_result(number_of_constraints_model=len(independence_constraints))
        refute.add_conditional_independence_test_result(
            number_of_constraints_model=len(independence_constraints),
            number_of_constraints_satisfied=len(self._true_implications),
            refutation_result=self._refutation_passed,
        )
        return refute


class GraphRefutation(CausalRefutation):
    """Class for storing the result of a refutation method."""

    def __init__(self, method_name_discrete, method_name_continuous):
        self.method_name_discrete = method_name_discrete
        self.method_name_continuous = method_name_continuous
        self.number_of_constraints_model = None
        self.number_of_constraints_satisfied = None
        self.refutation_result = None

    def add_conditional_independence_test_result(
        self, number_of_constraints_model, number_of_constraints_satisfied, refutation_result
    ):
        self.number_of_constraints_model = number_of_constraints_model
        self.number_of_constraints_satisfied = number_of_constraints_satisfied
        self.refutation_result = refutation_result

    def __str__(self):
        if self.refutation_result is None:
            return "Method name for discrete data:{0}\nMethod name for continuous data:{1}".format(
                self.method_name_discrete, self.method_name_continuous
            )
        else:
            return "Method name for discrete data:{0}\nMethod name for continuous data:{1}\nNumber of conditional independencies entailed by model:{2}\nNumber of independences satisfied by data:{3}\nTest passed:{4}\n".format(
                self.method_name_discrete,
                self.method_name_continuous,
                self.number_of_constraints_model,
                self.number_of_constraints_satisfied,
                self.refutation_result,
            )
