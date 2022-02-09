import logging
from dowhy.causal_refuter import CausalRefuter
from dowhy.utils.cit import partial_corr

class GraphRefuter(CausalRefuter):
    """
    Class for performing refutations on graph and storing the results
    """
    def __init__(self, data, method_name):
        """
        Initialize data for graph refutation

        :param data:input dataset
        :param method_name: name of method for testing conditional independence

        :returns : instance of GraphRefutation class
        """
        self._refutation_passed = False  
        self._data = data
        self._method_name = method_name
        self._false_implications = [] #List containing the implications from the graph which hold false for dataset
        self._true_implications = []  #List containing the implications from the graph which hold true for dataset
        self._results = {} #A dictionary with key as test set and value as [p-value, test_result]
        self.logger = logging.getLogger(__name__)

    def set_refutation_result(self):
        """
        Method to set the result for graph refutation. Set true if there are no false implications else false
        """
        if len(self._false_implications) == 0:
            self._refutation_passed = True
        else:
            self._refutation_passed = False
    
    def perform_conditional_independence_test(self,independence_constraints):
        """
        Method to test conditional independence using the graph refutation object on the given testing set

        :param independence_constraints: List of implications to test the conditional independence on
        """

        if self._method_name is None or self._method_name == "partial_correlation" :
            for a,b,c in independence_constraints:
                stats = partial_corr(data=self._data, x= a, y=b, z=list(c))
                p_value = stats['p-val']
                key = ((a,b)+(c,))
                if(p_value < 0.05):
                    #Reject H0
                    self._false_implications.append([a,b,c])
                    self._results[key]= [p_value, False]
                else:
                    self._true_implications.append([a, b, c])
                    self._results[key]= [p_value, True]
            self.set_refutation_result() #Set refutation result accordingly
        else:
            self.logger.error("Invalid conditional independence test")