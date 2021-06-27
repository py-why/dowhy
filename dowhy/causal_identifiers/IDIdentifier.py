from networkx.algorithms.assortativity.pairs import node_attribute_xy
import numpy as np
import pandas as pd
import networkx as nx
from ordered_set import OrderedSet

from dowhy.utils.graph_operations import find_c_components, induced_graph, find_ancestor

from dowhy.causal_identifier import CausalIdentifier
from dowhy.utils.api import parse_state

class IDExpression:

    def __init__(self):
        self._product = []
        self._sum = []
    
    def add_product(self, element):
        self._product.append(element)
        
    def add_sum(self, element):
        for el in element:
            self._sum.append(el)

    def get_val(self, return_type):
        """type = prod or sum"""
        if return_type=='prod':
            return self._product
        elif return_type=="sum":
            return self._sum
        else:
            raise Exception("Provide correct return type.")

    def _print_estimator(self, prefix, estimator=None):
        if estimator is None:
            estimator = self

        string = ""
        if isinstance(estimator, IDExpression):
            s = True if len(estimator.get_val(return_type="sum"))>0 else False
            if s:
                sum_vars = "{"
                for i, var in enumerate(list(estimator.get_val(return_type="sum"))):
                    sum_vars += var
                    if i < len(list(estimator.get_val(return_type="sum")))-1:
                        sum_vars += ","
                string += prefix + "Sum over " + sum_vars + "}:\n"
                prefix += "\t"
            for expression in estimator.get_val(return_type='prod'):
                string += self._print_estimator(prefix, expression)
        else:
            string += prefix + "Predictor: P("
            outcome_vars = list(estimator['outcome_vars'])
            for i, var in enumerate(outcome_vars):
                string += var
                if i<len(outcome_vars)-1:
                    string += ","
            condition_vars = list(estimator['condition_vars'])
            if len(condition_vars)>0:
                string += "|"
                for i, var in enumerate(condition_vars):
                    string += var
                    if i<len(condition_vars)-1:
                        string += ","
            string += ")\n"
        return string

    def __str__(self):
        return self._print_estimator(prefix="")

class IDIdentifier(CausalIdentifier):

    def __init__(self, graph, estimand_type,
            method_name = "default",
            proceed_when_unidentifiable=False):
        '''
        Class to perform identification using the ID algorithm.

        :param self: instance of the IDIdentifier class.
        :param treatment_names: list of treatment variables.
        :param outcome_names: list of outcome variables.
        :param causal_model: A CausalModel object.
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
        
        # # Estimators list for returning after identification
        # self._estimators = IDExpression()

    def identify_effect(self, treatment_names=None, outcome_names=None, adjacency_matrix=None, node_names=None):
        if adjacency_matrix is None:
            adjacency_matrix = self._adjacency_matrix
        if treatment_names is None:
            treatment_names = self._treatment_names
        if outcome_names is None:
            outcome_names = self._outcome_names
        if node_names is None:
            node_names = self._node_names
        node2idx, idx2node = self._idx_node_mapping(node_names)
        estimators = IDExpression()

        # Line 1
        if len(treatment_names) == 0:
            identifier = IDExpression()
            estimator = {}
            estimator['outcome_vars'] = outcome_names
            estimator['condition_vars'] = OrderedSet()
            identifier.add_product(estimator)
            identifier.add_sum(node_names - outcome_names)
            # self._estimators.add_product(identifier)
            # return self._estimators
            estimators.add_product(identifier)
            return estimators


        # Line 2 - Remove ancestral nodes that don't affect output
        ancestors = find_ancestor(outcome_names, node_names, adjacency_matrix, node2idx, idx2node)
        if len(node_names - ancestors) != 0: # If there are elements which are not the ancestor of the outcome variables
            # Modify list of valid nodes
            treatment_names = treatment_names & ancestors
            node_names = node_names & ancestors
            adjacency_matrix = induced_graph(node_set=node_names, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
            return self.identify_effect(treatment_names=treatment_names, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 3
        # Modify adjacency matrix to obtain that corresponding to do(X)
        adjacency_matrix_do_x = adjacency_matrix.copy()
        for x in treatment_names:
            x_idx = node2idx[x]
            for i in range(len(node_names)):
                adjacency_matrix_do_x[i, x_idx] = 0
        ancestors = find_ancestor(outcome_names, node_names, adjacency_matrix_do_x, node2idx, idx2node)
        W = node_names - treatment_names - ancestors
        if len(W) != 0:
            return self.identify_effect(treatment_names = treatment_names | W, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 4
        # Modify adjacency matrix to remove treatment variables
        node_names_minus_x = node_names - treatment_names
        node2idx_minus_x, idx2node_minus_x = self._idx_node_mapping(node_names_minus_x)
        adjacency_matrix_minus_x = induced_graph(node_set=node_names_minus_x, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
        c_components = find_c_components(adjacency_matrix=adjacency_matrix_minus_x, node_set=node_names_minus_x, idx2node=idx2node_minus_x)
        if len(c_components)>1:
            identifier = IDExpression()
            sum_over_set = node_names - (outcome_names | treatment_names)
            for component in c_components:
                expressions = self.identify_effect(treatment_names=node_names-component, outcome_names=OrderedSet(list(component)), adjacency_matrix=adjacency_matrix, node_names=node_names)
                for expression in expressions.get_val(return_type="prod"):
                    identifier.add_product(expression)
            identifier.add_sum(sum_over_set)
            # self._estimators.add_product(identifier)
            # return self._estimators
            estimators.add_product(identifier)
            return estimators

        
        # Line 5
        S = c_components[0]
        c_components_G = find_c_components(adjacency_matrix=adjacency_matrix, node_set=node_names, idx2node=idx2node)
        if len(c_components_G)==1 and c_components_G[0] == node_names:
            return ["FAIL"]
    
        # Line 6
        if S in c_components_G:
            sum_over_set = S - outcome_names
            prev_nodes = []
            for node in self._tsort_node_names:
                if node in S:
                    identifier = IDExpression()
                    estimator = {}
                    estimator['outcome_vars'] = OrderedSet([node])
                    estimator['condition_vars'] = OrderedSet(prev_nodes)
                    identifier.add_product(estimator)
                    identifier.add_sum(sum_over_set)
                    # Check if estimator already added
                    # self._estimators.add_product(identifier)
                    estimators.add_product(identifier)
                prev_nodes.append(node)
            # return self._estimators
            return estimators


        # Line 7
        for component in c_components_G:
            if S - component is None:
                return self.identify_effect(treatment_names=treatment_names & component, outcome_names=outcome_names, adjacency_matrix=self._induced_graph(node_set=component, adjacency_matrix=adjacency_matrix,node2idx=node2idx), node_names=node_names)
    
    def _idx_node_mapping(self, node_names):
        node2idx = {}
        idx2node = {}
        for i, node in enumerate(node_names):
            node2idx[node] = i
            idx2node[i] = node
        return node2idx, idx2node