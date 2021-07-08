import networkx as nx
from dowhy.utils.graph_operations import adjacency_matrix_to_adjacency_list

class NodePair:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.is_blocked = None # To store if all paths between node1 and node2 are blocked
        self.condition_vars = set() # To store variable to be conditioned on to block all paths between node1 and node2

class Path:
    def __init__(self):
        self._path = list() # To store variables in the path
        self._is_blocked = None # To store if path is blocked
        self._condition_vars = set() # To store variables needed to block the path
    
    def update_path(self, path, is_blocked):
        self._path = path
        self._is_blocked = is_blocked
        if not is_blocked:
            self._condition_vars = self._condition_vars.union(set(path[1:-1]))
    
    def get_path(self):
        return self._path
    
    def __str__(self):
        print("Path:", self._path)
        print("Blocked:", self._is_blocked)
        if not self._is_blocked:
            print("To block path, condition on:", self._condition_vars)

class Backdoor:

    def __init__(self, graph, nodes1, nodes2):
        self._graph = graph
        self._nodes1 = nodes1
        self._nodes2 = nodes2
    
    def get_backdoor_paths(self):
        undirected_graph = self._graph.to_undirected()
        nodes12 = set(self._nodes1).union(self._nodes2)
        
        # Get adjacency list
        adjlist = adjacency_matrix_to_adjacency_list(nx.to_numpy_matrix(undirected_graph), labels=list(undirected_graph.nodes))
        path_dict = {}

        for node1 in self._nodes1:
            for node2 in self._nodes2:
                if (node1, node2) in path_dict:
                    continue

                paths = self._path_search(adjlist, node1, node2)
                backdoor_paths = [
                    pth
                    for pth in paths
                    if self._graph.has_edge(pth.get_path()[1], pth.get_path()[0])]
                filtered_backdoor_paths = [
                    pth
                    for pth in backdoor_paths
                    if len(nodes12.intersection(set(pth.get_path()[1:-1])))==0]
                path_dict[(node1, node2)] = filtered_backdoor_paths
            
        return path_dict

    def _path_search_util(self, graph, node1, node2, vis, path, paths, is_blocked=False):
        path.append(node1)
        vis.add(node1)
        if node1 == node2:
            path_var = Path()
            path_var.update_path(path.copy(), is_blocked)
            paths.append(path_var)
        else:
            for neighbour in graph[node1]:
                if neighbour not in vis:
                    self._path_search_util(graph, neighbour, node2, vis, path, paths)
        path.pop()
        vis.remove(node1)
        return paths

    def _path_search(self, graph, node1, node2):
        '''
        Path search using DFS.
        '''
        vis = set()
        paths = self._path_search_util(graph, node1, node2, vis, [], [])
        return paths
