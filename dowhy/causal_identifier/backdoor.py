import networkx as nx

from dowhy.causal_identifier.adjustment_set import AdjustmentSet
from dowhy.utils.graph_operations import adjacency_matrix_to_adjacency_list


class NodePair:
    """
    Data structure to store backdoor variables between 2 nodes.
    """

    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2
        self._is_blocked = None  # To store if all paths between node1 and node2 are blocked
        self._condition_vars = []  # To store variable to be conditioned on to block all paths between node1 and node2
        self._complete = False  # To store to paths between node pair have been completely explored.

    def update(self, path, condition_vars=None):
        if condition_vars is None:
            """path is a Path variable"""
            if self._is_blocked is None:
                self._is_blocked = path.is_blocked()
            else:
                self._is_blocked = self._is_blocked and path.is_blocked()
            if not path.is_blocked():
                self._condition_vars.append(path.get_condition_vars())

        else:
            """path is a list"""
            condition_vars = list(condition_vars)
            self._condition_vars.append(set([*path[1:], *condition_vars]))

    def get_condition_vars(self):
        return self._condition_vars

    def set_complete(self):
        self._complete = True

    def is_complete(self):
        return self._complete

    def __str__(self):
        string = ""
        string += "Blocked: " + str(self._is_blocked) + "\n"
        if not self._is_blocked:
            condition_vars = [str(s) for s in self._condition_vars]
            string += "To block path, condition on: " + ",".join(condition_vars) + "\n"
        return string


class Path:
    """
    Data structure to store a particular path between 2 nodes.
    """

    def __init__(self):
        self._is_blocked = None  # To store if path is blocked
        self._condition_vars = set()  # To store variables needed to block the path

    def update(self, path, is_blocked):
        """
        path is a list
        """
        self._is_blocked = is_blocked
        if not is_blocked:
            self._condition_vars = self._condition_vars.union(set(path[1:-1]))

    def is_blocked(self):
        return self._is_blocked

    def get_condition_vars(self):
        return self._condition_vars

    def __str__(self):
        string = ""
        string += "Blocked: " + str(self._is_blocked) + "\n"
        if not self._is_blocked:
            string += "To block path, condition on: " + ",".join(self._condition_vars) + "\n"
        return string


class Backdoor:
    """
    Class for optimized implementation of Backdoor variable search between the source nodes and the target nodes.
    """

    def __init__(self, graph, nodes1, nodes2):
        self._graph = graph
        self._nodes1 = nodes1
        self._nodes2 = nodes2
        self._nodes12 = set(self._nodes1).union(self._nodes2)  # Total set of nodes
        self._colliders = set()

    def get_backdoor_vars(self):
        """
        Obtains sets of backdoor variable to condition on for each node pair.

        :returns:  List of sets with each set containing backdoor variable corresponding to a given node pair.
        """
        undirected_graph = self._graph.to_undirected()

        # Get adjacency list
        adjlist = adjacency_matrix_to_adjacency_list(
            nx.to_numpy_array(undirected_graph), labels=list(undirected_graph.nodes)
        )
        path_dict = {}
        backdoor_sets = []  # Put in backdoor sets format

        for node1 in self._nodes1:
            for node2 in self._nodes2:
                if (node1, node2) in path_dict:
                    continue
                self._path_search(adjlist, node1, node2, path_dict)
                if len(path_dict) != 0:
                    obj = HittingSetAlgorithm(path_dict[(node1, node2)].get_condition_vars(), self._colliders)
                    backdoor_sets.append(
                        AdjustmentSet(
                            adjustment_type=AdjustmentSet.BACKDOOR,
                            adjustment_variables=tuple(obj.find_set()),
                            num_paths_blocked_by_observed_nodes=obj.num_sets(),
                        )
                    )

        return backdoor_sets

    def is_backdoor(self, path):
        """
        Check if path is a backdoor path.

        :param path: List of nodes comprising the path.
        """
        if len(path) < 2:
            return False
        return True if self._graph.has_edge(path[1], path[0]) else False

    def _path_search_util(self, graph, node1, node2, vis, path, path_dict, is_blocked=False, prev_arrow=None):
        """
        :param graph: Adjacency list of the graph under consideration.
        :param node1: Current node being considered.
        :param node2: Target node.
        :param vis: Set of already visited nodes.
        :param path: List of nodes comprising the path upto node1.
        :path path_dict: Dictionary of node pairs.
        :param is_blocked: True is path is blocked by a collider, else False.
        :param prev_arrow: Described state of previous arrow. True if arrow incoming, False if arrow outgoing.
        """
        if is_blocked:
            return

        # If node pair has been fully explored
        if ((node1, node2) in path_dict) and (path_dict[(node1, node2)].is_complete()):
            for i in range(len(path)):
                if (path[i], node2) not in path_dict:
                    path_dict[(path[i], node2)] = NodePair(path[i], node2)
                obj = HittingSetAlgorithm(path_dict[(node1, node2)].get_condition_vars(), self._colliders)
                # Add node1 to backdoor set of node_pair
                s = set([node1])
                s = s.union(obj.find_set())
                path_dict[(path[i], node2)].update(path[i:], s)

        else:
            path.append(node1)
            vis.add(node1)
            if node1 == node2:
                # Check if path is backdoor and does not have nodes1\node1 or nodes2\node2 as intermediate nodes
                if self.is_backdoor(path) and len(self._nodes12.intersection(set(path[1:-1]))) == 0:
                    for i in range(len(path) - 1):
                        if (path[i], node2) not in path_dict:
                            path_dict[(path[i], node2)] = NodePair(path[i], node2)
                        path_var = Path()
                        path_var.update(path[i:].copy(), is_blocked)
                        path_dict[(path[i], node2)].update(path_var)
            else:
                for neighbour in graph[node1]:
                    if neighbour not in vis:
                        # True if arrow incoming, False if arrow outgoing
                        next_arrow = False if self._graph.has_edge(node1, neighbour) else True
                        if next_arrow == True and prev_arrow == True:
                            is_blocked = True
                            self._colliders.add(node1)
                        self._path_search_util(
                            graph, neighbour, node2, vis, path, path_dict, is_blocked, not next_arrow
                        )  # Since incoming for current node is outgoing for the next
            path.pop()
            vis.remove(node1)

        # Mark pair (node1, node2) complete
        if (node1, node2) in path_dict:
            path_dict[(node1, node2)].set_complete()

    def _path_search(self, graph, node1, node2, path_dict):
        """
        Path search using DFS.

        :param graph: Adjacency list of the graph under consideration.
        :param node1: Current node being considered.
        :param node2: Target node.
        :path path_dict: Dictionary of node pairs.
        """
        vis = set()
        self._path_search_util(graph, node1, node2, vis, [], path_dict, is_blocked=False)


class HittingSetAlgorithm:
    """
    Class for the Hitting Set Algorithm to obtain a approximate minimal set of backdoor variables
    to condition on for each node pair.
    """

    def __init__(self, list_of_sets, colliders=set()):
        """
        :param list_of_sets: List of sets such that each set comprises nodes representing a single backdoor path between a source node and a target node.
        """
        self._list_of_sets = list_of_sets
        self._colliders = colliders
        self._var_count = self._count_vars()

    def num_sets(self):
        """
        Obtain number of backdoor paths between a node pair.
        """
        return len(self._list_of_sets)

    def find_set(self):
        """
        Find approximate minimal set of nodes such that there is atleast one node from each set in list_of_sets.

        :returns: Approximate minimal set of nodes.
        """
        var_set = set()
        num_indices = len(self._list_of_sets)
        indices_covered = set()
        all_set_indices = set([i for i in range(num_indices)])

        while not self._is_covered(indices_covered, num_indices):
            set_index = all_set_indices - indices_covered
            max_el = self._max_occurence_var(var_dict=self._var_count)
            if max_el is None:
                break
            var_set.add(max_el)

            # Modify variable count and indices covered
            covered_present = self._indices_covered(el=max_el, set_index=set_index)
            self._modify_count(covered_present)
            indices_covered = indices_covered.union(covered_present)
        return var_set

    def _count_vars(self, set_index=None):
        """
        Obtain count of number of sets each particular node belongs to.

        :param set_index: Set of indices to consider for calculating the number of sets "hit" by a variable..
        """
        var_dict = {}

        if set_index == None:
            set_index = set([i for i in range(len(self._list_of_sets))])

        for idx in set_index:
            s = self._list_of_sets[idx]

            for el in s:
                if el not in self._colliders:
                    if el not in var_dict:
                        var_dict[el] = 0
                    var_dict[el] += 1

        return var_dict

    def _modify_count(self, indices_covered):
        """
        Modify count of number of sets each particular node belongs to based on nodes already covered in the previous iteration of the algorithm.
        """
        for idx in indices_covered:
            for el in self._list_of_sets[idx]:
                if el not in self._colliders:
                    self._var_count[el] -= 1

    def _max_occurence_var(self, var_dict):
        """
        Find the node contained in most number of sets.
        """
        max_el = None
        max_count = 0
        for key, val in var_dict.items():
            if val > max_count:
                max_count = val
                max_el = key
        return max_el

    def _indices_covered(self, el, set_index=None):
        """
        Obtain indices covered in a particular iteration of the algorithm.
        """
        covered = set()
        if set_index == None:
            set_index = set([i for i in range(len(self._list_of_sets))])

        for idx in set_index:
            if el in self._list_of_sets[idx]:
                covered.add(idx)
        return covered

    def _is_covered(self, indices_covered, num_indices):
        """
        List of sets is covered by the variable set.
        """

        covered = [False for i in range(num_indices)]
        for idx in indices_covered:
            covered[idx] = True
        return all(covered)
