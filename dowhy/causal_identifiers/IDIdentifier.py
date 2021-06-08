from networkx.algorithms.assortativity.pairs import node_attribute_xy
import numpy as np
import pandas as pd
import networkx as nx
from queue import LifoQueue
from ordered_set import OrderedSet

from dowhy.utils.api import parse_state

class IDIdentifier:

    def __init__(self, treatment_names=None, outcome_names=None, causal_model=None):# adjacency_matrix=None, node_names=None):
        '''
        Class to perform identification using the ID algorithm.

        :param self: instance of the IDIdentifier class.
        :param treatment_names: list of treatment variables.
        :param outcome_names: list of outcome variables.
        :param graph: A CausalGraph object.
        '''
        
        self._treatment_names = OrderedSet(parse_state(treatment_names))
        self._outcome_names = OrderedSet(parse_state(outcome_names))
        
        if causal_model is None:
            raise Exception("A CausalModel object must be provided for ID identification algorithm.")
        else:
            self._adjacency_matrix = causal_model._graph.get_adjacency_matrix()
            self._tsort_node_names = OrderedSet(list(nx.topological_sort(causal_model._graph._graph))) # topological sorting of graph nodes
            self._node_names = OrderedSet(causal_model._graph._graph.nodes)
            print("Original Ordering of nodes:", causal_model._graph._graph.nodes)
            print("Current ordering of nodes:", self._node_names)
            print("Adjacency Matrix:")
            print(self._adjacency_matrix)
            
        # Estimators list for returning after identification
        self._estimators = []

    def identify(self, treatment_names=None, outcome_names=None, adjacency_matrix=None, node_names=None):
        if adjacency_matrix is None:
            adjacency_matrix = self._adjacency_matrix
        if treatment_names is None:
            treatment_names = self._treatment_names
        if outcome_names is None:
            outcome_names = self._outcome_names
        if node_names is None:
            node_names = self._node_names
        node2idx, idx2node = self._idx_node_mapping(node_names)
        print("####################################################################################")
        print(treatment_names, outcome_names, node_names)
        print(node2idx, idx2node)

        # Line 1
        print("Line 1")
        if len(treatment_names) == 0:
            print("Line 1 if")
            estimator = {}
            estimator['outcome_vars'] = outcome_names
            estimator['condition_vars'] = OrderedSet()
            estimator['marginalize_vars'] = node_names - outcome_names
            self._estimators.append(estimator)
            return self._estimators
        
        # Line 2 - Remove ancestral nodes that don't affect output
        print("Line 2")
        ancestors = self._find_ancestor(outcome_names, node_names, adjacency_matrix, node2idx, idx2node)
        print("Line 2 Ancestors:", ancestors)
        print("Line 2 Adj Matrix Shape:", adjacency_matrix.shape)
        if len(node_names - ancestors) != 0: # If there are elements which are not the ancestor of the outcome variables
            print("Line 2 if")
            # Modify list of valid nodes
            # set_wo_treatment = self._node_names - treatment_names
            # treatment_names = list(set(treatment_names).intersection(set(ancestors)))
            treatment_names = treatment_names & ancestors
            # node_names = set_wo_treatment | treatment_names
            # for i, node in enumerate(node_names):
            node2idx, idx2node = self._idx_node_mapping(ancestors)
            adjacency_matrix = self._induced_graph(node_set=ancestors, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
            return self.identify(treatment_names=treatment_names, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=ancestors)
            # return self.identify(treatment_names=treatment_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
        
        # Line 3
        print("Line 3")
        # Modify adjacency matrix to obtain that corresponding to do(X)
        adjacency_matrix_do_x = adjacency_matrix.copy()
        for x in treatment_names:
            x_idx = node2idx[x]
            for i in range(len(node_names)):
                adjacency_matrix_do_x[i, x_idx] = 0
        ancestors = self._find_ancestor(outcome_names, node_names, adjacency_matrix_do_x, node2idx, idx2node)
        W = node_names - treatment_names - ancestors
        if len(W) != 0:
            print("Line 3 if")
            return self.identify(treatment_names = treatment_names | W, outcome_names=outcome_names, adjacency_matrix=adjacency_matrix, node_names=node_names)
    
        # Line 4
        print("Line 4")
        # Modify adjacency matrix to remove treatment variables
        adjacency_matrix_minus_x = self._induced_graph(node_set=node_names-treatment_names, adjacency_matrix=adjacency_matrix, node2idx=node2idx)
        c_components = self._find_c_components(adjacency_matrix_minus_x, node_set=node_names-treatment_names, idx2node=idx2node)
        print("C Components:", c_components)
        # TODO: Take care of adding over v\(y union x)
        if len(c_components)>1:
            print("Line 4 if")
            estimators_list = []
            for component in c_components:
                estimators_list.append(self.identify(treatment_names=node_names-component, outcome_names=OrderedSet(list(component)), adjacency_matrix=adjacency_matrix, node_names=node_names))
            # estimators_list = sum(estimators_list, []) # Convert 2D list to 1D list
            estimators_list = [j for sub in estimators_list for j in sub] # Convert 2D list to 1D list
            sum_over_set = node_names - (outcome_names | treatment_names)
            print("Line 4 print:", sum_over_set, treatment_names, outcome_names, node_names)
            print("Line 4 estimators list:", estimators_list)
            for estimator in estimators_list:
                estimator['marginalize_vars'] |= sum_over_set
                self._estimators.append(estimator)
            return self._estimators
        
        # Line 5
        print("Line 5")
        S = c_components[0]
        c_components_G = self._find_c_components(adjacency_matrix, node_set=node_names, idx2node=idx2node)
        if len(c_components_G)==1 and c_components_G[0] == node_names:
            print("Line 5 if")
            return ["FAIL"]

        # Line 6
        if S in c_components_G:
            # Obtain topological ordering of nodes in S
            topological_order = []
            for node in self._tsort_node_names:
                if node in S:
                    topological_order.append(node)
            
            sum_over_set = S - outcome_names
            prev_nodes = []
            for node in topological_order:
                estimator = {}
                estimator['outcome_vars'] = OrderedSet(node)
                estimator['condition_vars'] = OrderedSet(prev_nodes)
                estimator['marginalize_vars'] = sum_over_set
                self._estimators.append(estimator)
                prev_nodes.append(node)
            return self._estimators
            # print("**************************************************")
            # print(self._tsort_node_names)
            # print(S, node_names, c_components_G)
            # exit()

        # Line 7
        for component in c_components_G:
            if S - component is None:
                return self.identify(treatment_names=treatment_names & component, outcome_names=outcome_names, adjacency_matrix=self._induced_graph(node_set=component, adjacency_matrix=adjacency_matrix,node2idx=node2idx), node_names=node_names)

        # return []

    def _find_ancestor(self, node_set, node_names, adjacency_matrix, node2idx, idx2node):
        ancestors = OrderedSet()
        for node_name in node_set:
            a = self._find_ancestor_help(node_name, node_names, adjacency_matrix, node2idx, idx2node)
            ancestors |= a
        return ancestors

    def _find_ancestor_help(self, node_name, node_names, adjacency_matrix, node2idx, idx2node):
        ancestors = OrderedSet()
        nodes_to_visit = LifoQueue(maxsize = len(self._node_names))
        # print(nodes_to_visit, self._node2idx[node_name])
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

        def dfs(node_idx, component):
            visited[node_idx] = True
            component.add(idx2node[node_idx])
            for neighbour in adjacency_list[node_idx]:
                if visited[neighbour] == False:
                    dfs(neighbour)

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
