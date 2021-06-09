from networkx.algorithms.assortativity.pairs import node_attribute_xy
import numpy as np
import pandas as pd
import networkx as nx
from queue import LifoQueue
from ordered_set import OrderedSet

from dowhy.causal_identifier import CausalIdentifier
from dowhy.utils.api import parse_state

class IDIdentifier: #(CausalIdentifier):

    def __init__(self, treatment_names=None, outcome_names=None, causal_model=None):
        '''
        Class to perform identification using the ID algorithm.

        :param self: instance of the IDIdentifier class.
        :param treatment_names: list of treatment variables.
        :param outcome_names: list of outcome variables.
        :param causal_model: A CausalModel object.
        '''

        # super().__init__(treatment_names=None, outcome_names=None, causal_model=None)
        
        self._treatment_names = OrderedSet(parse_state(treatment_names))
        self._outcome_names = OrderedSet(parse_state(outcome_names))
        
        if causal_model is None:
            raise Exception("A CausalModel object must be provided for ID identification algorithm.")
        else:
            self._adjacency_matrix = causal_model._graph.get_adjacency_matrix()
            self._tsort_node_names = OrderedSet(list(nx.topological_sort(causal_model._graph._graph))) # topological sorting of graph nodes
            self._node_names = OrderedSet(causal_model._graph._graph.nodes)
            # print("Original Ordering of nodes:", causal_model._graph._graph.nodes)
            # print("Current ordering of nodes:", self._node_names)
            # print("Adjacency Matrix:")
            # print(self._adjacency_matrix)
        
        # Estimators list for returning after identification
        self._estimators = []
        self._estimator_set = set()

    def identify_effect(self, treatment_names=None, outcome_names=None, adjacency_matrix=None, node_names=None):
        # print("Before Adjacency Matrix:")
        # print(adjacency_matrix)
        if adjacency_matrix is None:
            adjacency_matrix = self._adjacency_matrix
        if treatment_names is None:
            treatment_names = self._treatment_names
        if outcome_names is None:
            outcome_names = self._outcome_names
        if node_names is None:
            node_names = self._node_names
        node2idx, idx2node = self._idx_node_mapping(node_names)
        # print("####################################################################################")
        # print("Before Line 1:", treatment_names, outcome_names, node_names)
        # print("Adjacency Matrix:")
        # print(adjacency_matrix)
        # print(node2idx, idx2node)

        # Line 1
        # print("Line 1")
        if len(treatment_names) == 0:
            # print("Line 1 if")
            estimator = {}
            estimator['outcome_vars'] = outcome_names
            estimator['condition_vars'] = OrderedSet()
            estimator['marginalize_vars'] = node_names - outcome_names
            # Check if estimator already added 
            if not self._is_present(estimator):
                self._estimators.append(estimator)
                # print("Line 1 estimator:", estimator)
            return self._estimators
        # exit()

        # Line 2 - Remove ancestral nodes that don't affect output
        # print("Line 2")
        ancestors = self._find_ancestor(outcome_names, node_names, adjacency_matrix, node2idx, idx2node)
        # print("Line 2 Ancestors:", ancestors)
        if len(node_names - ancestors) != 0: # If there are elements which are not the ancestor of the outcome variables
            # print("Line 2 if")
            # Modify list of valid nodes
            treatment_names = treatment_names & ancestors
            node_names = node_names & ancestors
            # node2idx, idx2node = self._idx_node_mapping(ancestors)
            adjacency_matrix = self._induced_graph(node_set=node_names, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
            return self.identify_effect(treatment_names=treatment_names, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 3
        # print("Line 3")
        # Modify adjacency matrix to obtain that corresponding to do(X)
        adjacency_matrix_do_x = adjacency_matrix.copy()
        for x in treatment_names:
            x_idx = node2idx[x]
            for i in range(len(node_names)):
                adjacency_matrix_do_x[i, x_idx] = 0
        ancestors = self._find_ancestor(outcome_names, node_names, adjacency_matrix_do_x, node2idx, idx2node)
        W = node_names - treatment_names - ancestors
        if len(W) != 0:
            # print("Line 3 if")
            return self.identify_effect(treatment_names = treatment_names | W, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 4
        # print("Line 4")
        # Modify adjacency matrix to remove treatment variables
        node_names_minus_x = node_names - treatment_names
        node2idx_minus_x, idx2node_minus_x = self._idx_node_mapping(node_names_minus_x)
        adjacency_matrix_minus_x = self._induced_graph(node_set=node_names_minus_x, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
        c_components = self._find_c_components(adjacency_matrix=adjacency_matrix_minus_x, node_set=node_names_minus_x, idx2node=idx2node_minus_x)
        # TODO: Take care of adding over v\(y union x)
        if len(c_components)>1:
            # print("Line 4 if")
            sum_over_set = node_names - (outcome_names | treatment_names)
            for component in c_components:
                # print("Line 4 if For")
                estimators = self.identify_effect(treatment_names=node_names-component, outcome_names=OrderedSet(list(component)), adjacency_matrix=adjacency_matrix, node_names=node_names)
                for estimator in estimators:
                    estimator['marginalize_vars'] |= sum_over_set
                    # Check if estimator already added 
                    if not self._is_present(estimator):
                        self._estimators.append(estimator)
                        # print("Line 4 estimator:", estimator)
            return self._estimators
        
        # Line 5
        # print("Line 5")
        S = c_components[0]
        c_components_G = self._find_c_components(adjacency_matrix=adjacency_matrix, node_set=node_names, idx2node=idx2node)
        if len(c_components_G)==1 and c_components_G[0] == node_names:
            # print("Line 5 if")
            return ["FAIL"]
    
        # Line 6
        # print("Line 6")
        if S in c_components_G:
            # print("Line 6 if")
            sum_over_set = S - outcome_names
            prev_nodes = []
            for node in self._tsort_node_names:
                if node in S:
                    # print("Line 6 estimator node:", node)
                    estimator = {}
                    estimator['outcome_vars'] = OrderedSet([node])
                    estimator['condition_vars'] = OrderedSet(prev_nodes)
                    estimator['marginalize_vars'] = sum_over_set
                    # print("Line 6 estimator before if:", estimator)
                    # Check if estimator already added 
                    if not self._is_present(estimator):
                        self._estimators.append(estimator)
                        # print("Line 6 estimator:", estimator)
                prev_nodes.append(node)
            return self._estimators

        # Line 7
        # print("Line 7")
        for component in c_components_G:
            if S - component is None:
                # print("Line 7 if")
                return self.identify_effect(treatment_names=treatment_names & component, outcome_names=outcome_names, adjacency_matrix=self._induced_graph(node_set=component, adjacency_matrix=adjacency_matrix,node2idx=node2idx), node_names=node_names)

    def _find_ancestor(self, node_set, node_names, adjacency_matrix, node2idx, idx2node):
        ancestors = OrderedSet()
        for node_name in node_set:
            ancestors |= self._find_ancestor_help(node_name, node_names, adjacency_matrix, node2idx, idx2node)
        return ancestors

    def _find_ancestor_help(self, node_name, node_names, adjacency_matrix, node2idx, idx2node):
        ancestors = OrderedSet()
        nodes_to_visit = LifoQueue(maxsize = len(self._node_names))
        nodes_to_visit.put(node2idx[node_name])
        while not nodes_to_visit.empty():
            child = nodes_to_visit.get()
            ancestors.add(idx2node[child])
            for i in range(len(node_names)):
                if idx2node[i] not in ancestors and adjacency_matrix[i, child] == 1: # For edge a->b, a is along height and b is along width of adjacency matrix
                    nodes_to_visit.put(i)
        return ancestors

    def _induced_graph(self, node_set, adjacency_matrix, node2idx):
        node_idx_list = [node2idx[node] for node in node_set]
        node_idx_list.sort()
        adjacency_matrix_induced = adjacency_matrix.copy()
        adjacency_matrix_induced = adjacency_matrix_induced[node_idx_list]
        adjacency_matrix_induced = adjacency_matrix_induced[:, node_idx_list]
        return adjacency_matrix_induced        

    def _find_c_components(self, adjacency_matrix, node_set, idx2node):

        if node_set is None:
            node_set = self._node_names
        num_nodes = len(node_set)
        adj_matrix = adjacency_matrix.copy()
        adjacency_list = [[] for _ in range(num_nodes)]

        # Modify graph such that it only contains bidirected edges
        for h in range(0, num_nodes-1):
            for w in range(h+1, num_nodes):
                if adjacency_matrix[h, w]==1 and adjacency_matrix[w, h]==1:
                    adjacency_list[h].append(w)
                    adjacency_list[w].append(h)
                else:
                    adj_matrix[h, w] = 0
                    adj_matrix[w, h] = 0

        # Find c components by finding connected components on the undirected graph
        visited = [False for _ in range(num_nodes)]

        def dfs(node_idx, component):
            visited[node_idx] = True
            component.add(idx2node[node_idx])
            for neighbour in adjacency_list[node_idx]:
                if visited[neighbour] == False:
                    dfs(neighbour, component)

        c_components = []
        for i in range(num_nodes):
            if visited[i] == False:
                component = OrderedSet()
                dfs(i, component)
                c_components.append(component)

        return c_components
    
    def _idx_node_mapping(self, node_names):
        node2idx = {}
        idx2node = {}
        for i, node in enumerate(node_names):
            node2idx[node] = i
            idx2node[i] = node
        return node2idx, idx2node

    def _is_present(self, estimator):
        string = ""
        for node in estimator['outcome_vars']:
            string += node
        string += "|"
        for node in estimator['condition_vars']:
            string += node
        string += "|"
        for node in estimator['marginalize_vars']:
            string += node
    
        if string not in self._estimator_set:
            self._estimator_set.add(string)
            return False
        return True
    
    