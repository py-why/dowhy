from networkx.algorithms.assortativity.pairs import node_attribute_xy
import numpy as np
import pandas as pd
from queue import LifoQueue

from dowhy.utils.api import parse_state

class IDIdentifier:

    def __init__(self, treatment_names=[], outcome_names=[], adjacency_matrix=None, node_names=None):
        '''
        Class to perform identification using the ID algorithm.

        :param self: instance of the IDIdentifier class.
        :param treatment: list of treatment variables.
        :param outcome: list of outcome variables.
        :param graph: A CausalGraph object.
        '''
        
        # TODO - Covert to lists if not provided as lists
        self._treatment_names = set(parse_state(treatment_names))
        self._outcome_names = set(parse_state(outcome_names))
        
        if adjacency_matrix is None:
            raise Exception("Graph must be provided for ID identification algorithm.")
        else:
            # self._adjacency_matrix = graph.get_adjacency_matrix()
            self._adjacency_matrix = adjacency_matrix
            if node_names == None:
                self._node_names = [str(i) for i in range(self._adjacency_matrix.shape[0])]
            else:
                # self._node_names = graph._graph.nodes
                self._node_names = node_names
            self._node2idx = {}
            self._idx2node = {}
            for i, node in enumerate(self._node_names):
                self._node2idx[node] = i
                self._idx2node[i] = node
            self._node_names = set(self._node_names)
        
        # Estimators list for returning after identification
        self._estimators = []

    def identify(self, treatment_names=None, outcome_names=None, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self._adjacency_matrix
        if treatment_names is None:
            treatment_names = self._treatment_names
        if outcome_names is None:
            outcome_names = self._outcome_names
        # print(adjacency_matrix, treatment_names, outcome_names)
        # print(self._node_names)
        # exit()
        # Line 1
        if len(treatment_names) == 0:
            estimator = {}
            estimator['condition_vars'] = set()
            estimator['marginalize_vars'] = self.node_names - outcome_names
            self._estimators.append(estimator)
            return self._estimators
        
        # Line 2 - Remove ancestral nodes that don't affect output
        ancestors = self._find_ancestor(outcome_names, adjacency_matrix)
        node_names = self._node_names.copy()
        if len(node_names - ancestors) != 0: # If there are elements which are not the ancestor of the outcome variables
            # Modify list of valid nodes
            set_wo_treatment = self._node_names - treatment_names
            treatment_names = treatment_names.intersection(ancestors)
            self._node_names = set_wo_treatment | treatment_names
            for i, node in enumerate(self._node_names):
                self._node2idx[node] = i
                self._idx2node[i] = node
            adjacency_matrix = self._induced_graph(self._node_names, adjacency_matrix)
            return self.identify(treatment_names=treatment_names, adjacency_matrix=adjacency_matrix)
        
        # Line 3
        # Modify adjacency matrix to obtain that corresponding to do(X)
        adjacency_matrix_do_x = adjacency_matrix.copy()
        for x in treatment_names:
            x_idx = self._node2idx[x]
            for i in range(len(self._node_names)):
                adjacency_matrix_do_x[i, x_idx] = 0
        ancestors = self._find_ancestor(outcome_names, adjacency_matrix_do_x)
        W = self._node_names - treatment_names - ancestors
        if len(W) != 0:
            return self.identify(treatment_names = treatment_names.union(W), adjacency_matrix=adjacency_matrix)


    def _find_ancestor(self, node_set, adjacency_matrix):
        ancestors = set()
        for node in node_set:
            a = self._find_ancestor_help(node, adjacency_matrix)
            ancestors = ancestors.union(a)
        return ancestors

    def _find_ancestor_help(self, node_name, adjacency_matrix):
        
        ancestors = set()
        
        nodes_to_visit = LifoQueue(maxsize = len(self._node_names))
        nodes_to_visit.put(self._node2idx[node_name])
        while not nodes_to_visit.empty():
            child = nodes_to_visit.get()
            ancestors.add(self._idx2node[child])
            for i in range(len(self._node_names)):
                # print(i, self._idx2node[i], child, self._idx2node[child])
                # print(self._idx2node[i])
                # print(self._idx2node[i] not in ancestors)
                # print(adjacency_matrix.shape, i, child)
                if self._idx2node[i] not in ancestors and adjacency_matrix[i, child] == 1: # For edge a->b, a is along height and b is along width of adjacency matrix
                    nodes_to_visit.put(i)
        
        return ancestors

    def _induced_graph(self, node_set, adjacency_matrix):
        node_idx_list = [self._node2idx[node] for node in node_set].sort()
        adjacency_matrix = adjacency_matrix[node_idx_list]
        adjacency_matrix = adjacency_matrix[:, node_idx_list]
        return adjacency_matrix        

