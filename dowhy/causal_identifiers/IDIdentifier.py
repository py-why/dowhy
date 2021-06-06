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
            estimator['marginalize_vars'] = self._node_names - outcome_names
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
        
        # Line 4
        # Modify adjacency matrix to remove treatment variables
        adjacency_matrix_minus_x = self._induced_graph(node_set=self._node_names-treatment_names, adjacency_matrix=adjacency_matrix)
        c_components = self._find_c_components(adjacency_matrix_minus_x, node_set=self._node_names-treatment_names)
        # TODO: Take care of adding over v\(y union x)
        if len(c_components)>1:
            estimators_list = []
            for component in c_components:
                estimators_list.append(self.identify(treatment_names=self._node_names-component, outcome_names=component, adjacency_matrix=adjacency_matrix))
            sum_over_set = self._node_names - outcome_names.union(treatment_names)
            for estimator in estimators_list:
                estimator['marginalize_vars'] = estimator['marginalize_vars'].union(sum_over_set)
                self._estimators.append(estimator)
            return self._estimators
        
        # Line 5
        S = list(c_components)[0]
        c_components_G = self._find_c_components(adjacency_matrix)
        if len(c_components_G)==1 and list(c_components_G)[0] == self._node_names:
            return "FAIL"

        # Line 6
        if S in c_components_G:
            pass

        # Line 7
        for component in c_components_G:
            if S - component is None:
                return identify(treatment_names=treatment_names.intersection(component), outcome_names=outcome_names, adjacency_matrix=self._induced_graph(node_set=component, adjacency_matrix=adjacency_matrix))

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
                if self._idx2node[i] not in ancestors and adjacency_matrix[i, child] == 1: # For edge a->b, a is along height and b is along width of adjacency matrix
                    nodes_to_visit.put(i)
        
        return ancestors

    def _induced_graph(self, node_set, adjacency_matrix):
        node_idx_list = [self._node2idx[node] for node in node_set]
        node_idx_list.sort()
        adjacency_matrix_induced = adjacency_matrix.copy()
        adjacency_matrix_induced = adjacency_matrix_induced[node_idx_list]
        adjacency_matrix_induced = adjacency_matrix_induced[:, node_idx_list]
        return adjacency_matrix_induced        

    def _find_c_components(self, adjacency_matrix, node_set=None):
        if node_set is None:
            node_set = self._node_names
        num_nodes = len(node_set)
        adjacency_list = [[] for _ in range(num_nodes)]

        # Modify graph such that it only contains bidirected edges
        for h in range(0, num_nodes-1):
            for w in range(h+1, num_nodes):
                if adjacency_matrix[h, w]==1 and adjacency_matrix[w, h]==1:
                    adjacency_list[h].append(w)
                    adjacency_list[w].append(h)
                else:
                    adjacency_matrix[h, w] = 0
                    adjacency_matrix[w, h] = 0

        # Find c components by finding connected components on the undirected graph
        visited = [False for _ in range(num_nodes)]

        def dfs(node, component):
            visited[node] = True
            component.add(node)
            for neighbour in adjacency_list[node]:
                if visited[neighbour] == False:
                    dfs(neighbour)

        c_components = []
        for i in range(num_nodes):
            if visited[i] == False:
                component = set()
                dfs(i, component)
                c_components.append(component)

        return c_components


