import numpy as np
import pandas as pd
import networkx as nx
from dowhy.utils.ordered_set import OrderedSet
from dowhy.utils.graph_operations import find_c_components, induced_graph, find_ancestor
from dowhy.causal_identifier import CausalIdentifier
from dowhy.utils.api import parse_state

class IDExpression:
    """
    Class for storing a causal estimand, as a result of the identification step using the ID algorithm.
    The object stores a list of estimators(self._product) whose porduct must be obtained and a list of variables (self._sum) over which the product must be marginalized.
    """
    def __init__(self):
        self._product = []
        self._sum = []
    
    def add_product(self, element):
        '''
        Add an estimator to the list of product.

        :param element: Estimator to append to the product list.
        '''
        self._product.append(element)
        
    def add_sum(self, element):
        '''
        Add variables to the list.

        :param element: Set of variables to append to the list self._sum.
        '''
        for el in element:
            self._sum.append(el)

    def get_val(self, return_type):
        """
        Get either the list of estimators (for product) or list of variables (for the marginalization).

        :param return_type: "prod" to return the list of estimators or "sum" to return the list of variables.
        """
        if return_type=="prod":
            return self._product
        elif return_type=="sum":
            return self._sum
        else:
            raise Exception("Provide correct return type.")

    def _print_estimator(self, prefix, estimator=None, start=False):
        '''
        Print the IDExpression object.
        '''
        if estimator is None:
            return None
            
        string = ""
        if isinstance(estimator, IDExpression):
            s = True if len(estimator.get_val(return_type="sum"))>0 else False
            if s:
                sum_vars = "{" + ",".join(estimator.get_val(return_type="sum")) + "}"
                string += prefix + "Sum over " + sum_vars + ":\n"
                prefix += "\t"
            for expression in estimator.get_val(return_type='prod'):
                add_string = self._print_estimator(prefix, expression)
                if add_string is None:
                    return None
                else:
                    string += add_string
        else:
            outcome_vars = list(estimator['outcome_vars'])
            condition_vars = list(estimator['condition_vars'])
            string += prefix + "Predictor: P(" + ",".join(outcome_vars)
            if len(condition_vars)>0:
                string += "|" + ",".join(condition_vars)
            string += ")\n"
        if start:
            string = string[:-1]
        return string

    def __str__(self):
        string = self._print_estimator(prefix="", estimator=self, start=True)
        if string is None:
            return "The graph is not identifiable."
        else:
            return string

class IDIdentifier(CausalIdentifier):

    def __init__(self, graph, estimand_type,
            method_name = "default",
            proceed_when_unidentifiable=None):
        '''
        Class to perform identification using the ID algorithm.

        :param self: instance of the IDIdentifier class.
        :param estimand_type: Type of estimand ("nonparametric-ate", "nonparametric-nde" or "nonparametric-nie").
        :param method_name: Identification method ("id-algorithm" in this case).
        :param proceed_when_unidentifiable: If True, proceed with identification even in the presence of unobserved/missing variables.
        '''

        super().__init__(graph, estimand_type, method_name, proceed_when_unidentifiable)

        if self.estimand_type != CausalIdentifier.NONPARAMETRIC_ATE:
            raise Exception("The estimand type should be 'non-parametric ate' for the ID method type.")

        self._treatment_names = OrderedSet(parse_state(graph.treatment_name))
        self._outcome_names = OrderedSet(parse_state(graph.outcome_name))
        self._adjacency_matrix = graph.get_adjacency_matrix()

        try:
            self._tsort_node_names = OrderedSet(list(nx.topological_sort(graph._graph))) # topological sorting of graph nodes
        except:
            raise Exception("The graph must be a directed acyclic graph (DAG).")
        self._node_names = OrderedSet(graph._graph.nodes)
        
    def identify_effect(self, treatment_names=None, outcome_names=None, adjacency_matrix=None, node_names=None):
        '''
        Implementation of the ID algorithm.
        Link - https://ftp.cs.ucla.edu/pub/stat_ser/shpitser-thesis.pdf
        The pseudo code has been provided on Pg 40.

        :param self: instance of the IDIdentifier class.
        :param treatment_names: OrderedSet comprising names of treatment variables.
        :param outcome_names:OrderedSet comprising names of outcome variables.
        :param adjacency_matrix: Graph adjacency matrix.
        :param node_names: OrderedSet comprising names of all nodes in the graph.
        :returns:  target estimand, an instance of the IDExpression class.
        '''
        if adjacency_matrix is None:
            adjacency_matrix = self._adjacency_matrix
        if treatment_names is None:
            treatment_names = self._treatment_names
        if outcome_names is None:
            outcome_names = self._outcome_names
        if node_names is None:
            node_names = self._node_names
        node2idx, idx2node = self._idx_node_mapping(node_names)
        
        # Estimators list for returning after identification
        estimators = IDExpression()

        # Line 1
        # If no action has been taken, the effect on Y is just the marginal of the observational distribution P(v) on Y. 
        if len(treatment_names) == 0:
            identifier = IDExpression()
            estimator = {}
            estimator['outcome_vars'] = node_names
            estimator['condition_vars'] = OrderedSet()
            identifier.add_product(estimator)
            identifier.add_sum(node_names.difference(outcome_names))
            estimators.add_product(identifier)
            return estimators

        # Line 2
        # If we are interested in the effect on Y, it is sufficient to restrict our attention on the parts of the model ancestral to Y.
        ancestors = find_ancestor(outcome_names, node_names, adjacency_matrix, node2idx, idx2node)
        if len(node_names.difference(ancestors)) != 0: # If there are elements which are not the ancestor of the outcome variables
            # Modify list of valid nodes
            treatment_names = treatment_names.intersection(ancestors)
            node_names = node_names.intersection(ancestors)
            adjacency_matrix = induced_graph(node_set=node_names, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
            return self.identify_effect(treatment_names=treatment_names, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 3 - forces an action on any node where such an action would have no effect on Y – assuming we already acted on X. 
        # Modify adjacency matrix to obtain that corresponding to do(X)
        adjacency_matrix_do_x = adjacency_matrix.copy()
        for x in treatment_names:
            x_idx = node2idx[x]
            for i in range(len(node_names)):
                adjacency_matrix_do_x[i, x_idx] = 0
        ancestors = find_ancestor(outcome_names, node_names, adjacency_matrix_do_x, node2idx, idx2node)
        W = node_names.difference(treatment_names).difference(ancestors)
        if len(W) != 0:
            return self.identify_effect(treatment_names = treatment_names.union(W), outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 4 - Decomposes the problem into a set of smaller problems using the key property of C-component factorization of causal models.
        # If the entire graph is a single C-component already, further problem decomposition is impossible, and we must provide base cases.
        # Modify adjacency matrix to remove treatment variables
        node_names_minus_x = node_names.difference(treatment_names)
        node2idx_minus_x, idx2node_minus_x = self._idx_node_mapping(node_names_minus_x)
        adjacency_matrix_minus_x = induced_graph(node_set=node_names_minus_x, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
        c_components = find_c_components(adjacency_matrix=adjacency_matrix_minus_x, node_set=node_names_minus_x, idx2node=idx2node_minus_x)
        if len(c_components)>1:
            identifier = IDExpression()
            sum_over_set = node_names.difference(outcome_names.union(treatment_names))
            for component in c_components:
                expressions = self.identify_effect(treatment_names=node_names.difference(component), outcome_names=OrderedSet(list(component)), adjacency_matrix=adjacency_matrix, node_names=node_names)
                for expression in expressions.get_val(return_type="prod"):
                    identifier.add_product(expression)
            identifier.add_sum(sum_over_set)
            estimators.add_product(identifier)
            return estimators

        
        # Line 5 - The algorithms fails due to the presence of a hedge - the graph G, and a subgraph S that does not contain any X nodes.
        S = c_components[0]
        c_components_G = find_c_components(adjacency_matrix=adjacency_matrix, node_set=node_names, idx2node=idx2node)
        if len(c_components_G)==1 and c_components_G[0] == node_names:
            return None
    
        # Line 6 - If there are no bidirected arcs from X to the other nodes in the current subproblem under consideration, then we can replace acting on X by conditioning, and thus solve the subproblem.
        if S in c_components_G:
            sum_over_set = S.difference(outcome_names)
            prev_nodes = []
            for node in self._tsort_node_names:
                if node in S:
                    identifier = IDExpression()
                    estimator = {}
                    estimator['outcome_vars'] = OrderedSet([node])
                    estimator['condition_vars'] = OrderedSet(prev_nodes)
                    identifier.add_product(estimator)
                    identifier.add_sum(sum_over_set)
                    estimators.add_product(identifier)
                prev_nodes.append(node)
            return estimators


        # Line 7 - This is the most complicated case in the algorithm. Explain in the second last paragraph on Pg 41 of the link provided in the docstring above.
        for component in c_components_G:
            C = S.difference(component)
            if C.is_empty() is None:
                return self.identify_effect(treatment_names=treatment_names.intersection(component), outcome_names=outcome_names, adjacency_matrix=induced_graph(node_set=component, adjacency_matrix=adjacency_matrix,node2idx=node2idx), node_names=node_names)
    
    def _idx_node_mapping(self, node_names):
        '''
        Obtain the node name to index and index to node name mappings.
        
        :param node_names: Name of all nodes in the graph.
        :return: node to index and index to node mappings.
        '''
        node2idx = {}
        idx2node = {}
        for i, node in enumerate(node_names.get_all()):
            node2idx[node] = i
            idx2node[i] = node
        return node2idx, idx2node