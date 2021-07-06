import networkx as nx

class NodePair:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.is_blocked = None # To store if all paths between node1 and node2 are blocked
        self.condition_vars = set() # To store variable to be conditioned on to block all paths between node1 and node2

class Path:
    def __init__(self):
        self.path_var = list() # To store variables in the path
        self.is_blocked = None # To store if path is blocked
        self.condition_vars = set() # To store variables needed to block the path

def DFSUtil(graph, node1, node2, vis, path, paths):
    path.append(node1)
    vis.add(node1)
    if node1 == node2:
        paths.append(path.copy())
    else:
        for neighbour in graph[node1]:
            if neighbour not in vis:
                DFSUtil(graph, neighbour, node2, vis, path, paths)
    path.pop()
    vis.remove(node1)

def DFS(graph, node1, node2):
    print("DFS")
    vis = set()
    paths = []
    DFSUtil(graph, node1, node2, vis, [], paths)
    return paths

def adjacency_matrix_to_adjacency_list(adjacency_matrix, nodes=None):
    adjlist = dict()
    if nodes is None:
        nodes = [str(i+1) for i in range(adjacency_matrix.shape[0])]
    for i in range(adjacency_matrix.shape[0]):
        adjlist[nodes[i]] = list()
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                adjlist[nodes[i]].append(nodes[j])
    return adjlist

def get_backdoor_paths(graph, nodes1, nodes2):
    paths = []
    graph = graph.to_directed()
    undirected_graph = graph.to_undirected()
    nodes12 = set(nodes1).union(nodes2)
    
    # Get adjacency list
    adjlist = adjacency_matrix_to_adjacency_list(nx.to_numpy_matrix(undirected_graph), nodes=list(undirected_graph.nodes))
    
    for node1 in nodes1:
        for node2 in nodes2:
            paths = DFS(adjlist, node1, node2)
            backdoor_paths = [
                pth
                for pth in paths
                if graph.has_edge(pth[1], pth[0])]
            filtered_backdoor_paths = [
                pth
                for pth in backdoor_paths
                if len(nodes12.intersection(set(pth[1:-1])))==0]
        