import networkx as nx
from dowhy.utils.graph_operations import adjacency_matrix_to_adjacency_list

class NodePair:
    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2
        self._is_blocked = None # To store if all paths between node1 and node2 are blocked
        self._condition_vars = set() # To store variable to be conditioned on to block all paths between node1 and node2
    
    def update(self, paths):
        self._is_blocked = all([path.is_blocked() for path in paths])
        if not self._is_blocked:
            for path in paths:
                self._condition_vars = self._condition_vars.union(path.get_condition_vars())
        
    def __str__(self):
        string = ""
        string += "Blocked: " + str(self._is_blocked) + "\n"
        if not self._is_blocked:
            string += "To block path, condition on: " + ",".join(self._condition_vars) + "\n"
        return string
    

class Path:
    def __init__(self):
        # self._path = list() # To store variables in the path
        self._is_blocked = None # To store if path is blocked
        self._condition_vars = set() # To store variables needed to block the path
    
    def update(self, path, is_blocked):
        # self._path = path
        self._is_blocked = is_blocked
        if not is_blocked:
            self._condition_vars = self._condition_vars.union(set(path[1:-1]))
    
    def is_blocked(self):
        return self._is_blocked

    # def get_path(self):
    #     return self._path
    
    def get_condition_vars(self):
        return self._condition_vars
    
    def __str__(self):
        string = ""
        # string += "Path: " + ",".join(self._path) + "\n"
        string += "Blocked: " + str(self._is_blocked) + "\n"
        if not self._is_blocked:
            string += "To block path, condition on: " + ",".join(self._condition_vars) + "\n"
        return string
        
class Backdoor:

    def __init__(self, graph, nodes1, nodes2):
        self._graph = graph
        self._nodes1 = nodes1
        self._nodes2 = nodes2
        self._nodes12 = set(self._nodes1).union(self._nodes2)
        
    def get_backdoor_paths(self):
        undirected_graph = self._graph.to_undirected()
        
        # Get adjacency list
        adjlist = adjacency_matrix_to_adjacency_list(nx.to_numpy_matrix(undirected_graph), labels=list(undirected_graph.nodes))
        path_dict = {}

        for node1 in self._nodes1:
            for node2 in self._nodes2:
                if (node1, node2) in path_dict:
                    continue
                node_pair = NodePair(node1, node2)
                paths = self._path_search(adjlist, node1, node2)
                node_pair.update(paths)
                path_dict[(node1, node2)] = node_pair
                # path_dict[(node1, node2)] = paths
        return path_dict

    def is_backdoor(self, path):
        return True if self._graph.has_edge(path[1], path[0]) else False

    def _path_search_util(self, graph, node1, node2, vis, path, paths, is_blocked=False, prev_arrow=None):
        '''
        :param prev_arrow: True if arrow incoming, False if arrow outgoing
        '''
        if is_blocked:
            return paths
        path.append(node1)
        vis.add(node1)
        if node1 == node2:
            # Check if path is backdoor and does not have nodes1\node1 or nodes2\node2 as intermediate nodes
            if self.is_backdoor(path) and len(self._nodes12.intersection(set(path[1:-1])))==0:
                path_var = Path()
                path_var.update(path.copy(), is_blocked)
                paths.append(path_var)
        else:
            for neighbour in graph[node1]:
                if neighbour not in vis:
                    # True if arrow incoming, False if arrow outgoing
                    next_arrow = False if self._graph.has_edge(node1, neighbour) else True 
                    if next_arrow == True and prev_arrow == True:
                        is_blocked = True
                    paths = self._path_search_util(graph, neighbour, node2, vis, path, paths, is_blocked, not next_arrow) # Since incoming for current node is outgoing for the next
        path.pop()
        vis.remove(node1)
        return paths

    def _path_search(self, graph, node1, node2):
        '''
        Path search using DFS.
        '''
        vis = set()
        paths = self._path_search_util(graph, node1, node2, vis, [], [], False)
        return paths
