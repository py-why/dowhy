import re
from queue import LifoQueue

import networkx as nx
import numpy as np
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.shortest_paths.generic import shortest_path

from dowhy.utils.ordered_set import OrderedSet


def adjacency_matrix_to_adjacency_list(adjacency_matrix, labels=None):
    """
    Convert the adjacency matrix of a graph to an adjacency list.

    :param adjacency_matrix: A numpy array representing the graph adjacency matrix.
    :param labels: List of labels.
    :returns: Adjacency list as a dictionary.
    """

    adjlist = dict()
    if labels is None:
        labels = [str(i + 1) for i in range(adjacency_matrix.shape[0])]
    for i in range(adjacency_matrix.shape[0]):
        adjlist[labels[i]] = list()
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                adjlist[labels[i]].append(labels[j])
    return adjlist


def adjacency_matrix_to_graph(adjacency_matrix, labels=None):
    """
    Convert a given graph adjacency matrix to DOT format.

    :param adjacency_matrix: A numpy array representing the graph adjacency matrix.
    :param labels: List of labels.
    :returns: Graph in DOT format.
    """
    import graphviz

    if adjacency_matrix.ndim != 2:
        raise ValueError("Adjacency matrix must have a dimension of 2.")

    if isinstance(adjacency_matrix, np.matrix):
        adjacency_matrix = np.asarray(adjacency_matrix)

    # Only consider edges have absolute edge weight > 0.01
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)

    d = graphviz.Digraph(engine="dot")

    if labels is None:
        labels = [f"x{i}" for i in range(len(adjacency_matrix))]

    for label in labels:
        d.node(label)

    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(labels[from_], labels[to], label=str(coef))

    return d


def str_to_dot(string):
    """
    Converts input string from graphviz library to valid DOT graph format.

    :param string: Graph in DOT format.
    :returns: DOT string converted to a suitable format for the DoWhy library.
    """
    graph = string.strip().replace("\n", ";").replace("\t", "")
    graph = graph[:9] + graph[10:-2] + graph[-1]  # Removing unnecessary characters from string
    return graph


def find_ancestor(node_set, node_names, adjacency_matrix, node2idx, idx2node):
    """
    Finds ancestors of a given set of nodes in a given graph.

    :param node_set: Set of nodes whos ancestors must be obtained.
    :param node_names: Name of all nodes in the graph.
    :param adjacency_matrix: Graph adjacency matrix.
    :param node2idx: A dictionary mapping node names to their row or column index in the adjacency matrix.
    :param idx2node: A dictionary mapping the row or column indices in the adjacency matrix to the corresponding node names.
    :returns: OrderedSet containing ancestors of all nodes in the node_set.
    """

    def find_ancestor_help(node_name, node_names, adjacency_matrix, node2idx, idx2node):
        ancestors = OrderedSet()
        nodes_to_visit = LifoQueue(maxsize=len(node_names))
        nodes_to_visit.put(node2idx[node_name])
        while not nodes_to_visit.empty():
            child = nodes_to_visit.get()
            ancestors.add(idx2node[child])
            for i in range(len(node_names)):
                if (
                    idx2node[i] not in ancestors and adjacency_matrix[i, child] == 1
                ):  # For edge a->b, a is along height and b is along width of adjacency matrix
                    nodes_to_visit.put(i)
        return ancestors

    ancestors = OrderedSet()
    for node_name in node_set.get_all():
        ancestors = ancestors.union(find_ancestor_help(node_name, node_names, adjacency_matrix, node2idx, idx2node))
    return ancestors


def induced_graph(node_set, adjacency_matrix, node2idx):
    """
    To obtain the induced graph corresponding to a subset of nodes.

    :param node_set: Set of nodes whos ancestors must be obtained.
    :param adjacency_matrix: Graph adjacency matrix.
    :param node2idx: A dictionary mapping node names to their row or column index in the adjacency matrix.
    :returns: Numpy array representing the adjacency matrix of the induced graph.
    """
    node_idx_list = [node2idx[node] for node in node_set]
    node_idx_list.sort()
    adjacency_matrix_induced = adjacency_matrix.copy()
    adjacency_matrix_induced = adjacency_matrix_induced[node_idx_list]
    adjacency_matrix_induced = adjacency_matrix_induced[:, node_idx_list]
    return adjacency_matrix_induced


def find_c_components(adjacency_matrix, node_set, idx2node):
    """
    Obtain C-components in a graph.

    :param adjacency_matrix: Graph adjacency matrix.
    :param node_set: Set of nodes whos ancestors must be obtained.
    :param idx2node: A dictionary mapping the row or column indices in the adjacency matrix to the corresponding node names.
    :returns: List of C-components in the graph.
    """
    num_nodes = len(node_set)
    adj_matrix = adjacency_matrix.copy()
    adjacency_list = [[] for _ in range(num_nodes)]

    # Modify graph such that it only contains bidirected edges
    for h in range(0, num_nodes - 1):
        for w in range(h + 1, num_nodes):
            if adjacency_matrix[h, w] == 1 and adjacency_matrix[w, h] == 1:
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


def daggity_to_dot(daggity_string):
    """
    Converts the input daggity_string to valid DOT graph format.

    :param daggity_string: Output graph from Daggity site
    :returns: DOT string
    """
    graph = re.sub(r"\n", "; ", daggity_string)
    graph = re.sub(r"^dag ", "digraph ", graph)
    graph = re.sub("{;", "{", graph)
    graph = re.sub("};", "}", graph)
    graph = re.sub("outcome,*,", "", graph)
    graph = re.sub("adjusted,*", "", graph)
    graph = re.sub("exposure,*", "", graph)
    graph = re.sub("latent,*", 'observed="no",', graph)
    graph = re.sub(",]", "]", graph)
    return graph


def get_simple_ordered_tree(n):
    """
    Generates a simple-ordered tree. The tree is just a
    directed acyclic graph of n nodes with the structure
    0 --> 1 --> .... --> n.
    """
    g = nx.DiGraph()

    for i in range(n):
        g.add_node(i)

    for i in range(n - 1):
        g.add_edges_from([(i, i + 1, {})])
    return g


def is_connected(g):
    """
    Checks if a the directed acyclic graph is connected.
    """
    u = convert_to_undirected_graph(g)
    return nx.is_connected(u)


def convert_to_undirected_graph(g):
    u = nx.Graph()
    for n in g.nodes:
        u.add_node(n)
    for e in g.edges:
        u.add_edges_from([(e[0], e[1], {})])
    return u


def get_random_node_pair(n):
    """
    Randomly generates a pair of nodes.
    """
    i = np.random.randint(0, n)
    j = i
    while j == i:
        j = np.random.randint(0, n)
    return i, j


def find_predecessor(i, j, g):
    """
    Finds a predecessor, k, in the path between two nodes, i and j,
    in the graph, g.
    """
    parents = list(g.predecessors(j))
    u = convert_to_undirected_graph(g)
    for pa in parents:
        try:
            path = shortest_path(u, pa, i)
            return pa
        except:
            pass
    return None


def del_edge(i, j, g):
    """
    Deletes the edge i --> j in the graph, g. The edge is only
    deleted if this removal does NOT cause the graph to be
    disconnected.
    """
    if g.has_edge(i, j) is True:
        g.remove_edge(i, j)

        if is_connected(g) is False:
            g.add_edges_from([(i, j, {})])


def add_edge(i, j, g):
    """
    Adds an edge i --> j to the graph, g. The edge is only
    added if this addition does NOT cause the graph to have
    cycles.
    """
    g.add_edges_from([(i, j, {})])
    if is_directed_acyclic_graph(g) is False:
        g.remove_edge(i, j)
