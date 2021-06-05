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

        # Line 1
        if len(treatment_names) == 0:
            estimator = {}
            estimator['condition_vars'] = set()
            estimator['marginalize_vars'] = self.node_names - outcome_names
            self._estimators.append(estimator)
            return self._estimators
        
        # Line 2 - Remove ancestral nodes that don't affect output
        ancestors = self._find_ancestor(outcome_names)
        if self._node_names.difference_update(ancestors) != None: # If there are elements which are not the ancestor of the outcome variables
            # Modify list of valid nodes
            set_wo_treatment = self._node_names.difference_update(treatment_names)
            treatment_names = treatment_names.intersection(ancestors)
            self._node_names = set_wo_treatment | treatment_names
            adjacency_matrix = self._induced_graph(self._node_names)
            return self.identify(adjacency_matrix=adjacency_matrix)
        
        # Line 3



    def _find_ancestor(self, node_set):
        ancestors = set()
        for node in node_set:
            a = self._find_ancestor_help(node)
            ancestors = ancestors.union(a)
        return ancestors

    def _find_ancestor_help(self, node_name):
        
        ancestors = set()
        
        nodes_to_visit = LifoQueue(maxsize = len(self._node_names))
        nodes_to_visit.put(self._node2idx[node_name])
        while not nodes_to_visit.empty():
            child = nodes_to_visit.get()
            ancestors.add(self._idx2node[child])
            for i in range(len(self._node_names)):
                if self._idx2node[i] not in ancestors and self._adjacency_matrix[i, child] == 1:
                    nodes_to_visit.put(i)
        
        return ancestors

    def _induced_graph(self, node_set):
        node_idx_list = [self._node2idx[node] for node in node_set].sort()
        adjacency_matrix = self._adjacency_matrix[node_idx_list]
        adjacency_matrix = adjacency_matrix[:, node_idx_list]
        return adjacency_matrix        

