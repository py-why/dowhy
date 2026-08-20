"""Tests for dowhy/utils/graph_operations.py utility functions."""

import numpy as np
import pytest

from dowhy.utils.graph_operations import (
    add_edge,
    adjacency_matrix_to_adjacency_list,
    convert_to_undirected_graph,
    daggity_to_dot,
    del_edge,
    find_ancestor,
    get_simple_ordered_tree,
    induced_graph,
    is_connected,
    str_to_dot,
)
from dowhy.utils.ordered_set import OrderedSet


class TestAdjacencyMatrixToAdjacencyList:
    def test_simple_chain(self):
        """0->1->2 encoded as adjacency matrix."""
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        result = adjacency_matrix_to_adjacency_list(adj)
        assert result == {"1": ["2"], "2": ["3"], "3": []}

    def test_custom_labels(self):
        adj = np.array([[0, 1], [0, 0]])
        result = adjacency_matrix_to_adjacency_list(adj, labels=["X", "Y"])
        assert result == {"X": ["Y"], "Y": []}

    def test_no_edges(self):
        adj = np.zeros((3, 3))
        result = adjacency_matrix_to_adjacency_list(adj)
        assert all(v == [] for v in result.values())

    def test_default_labels_are_one_indexed(self):
        adj = np.zeros((3, 3))
        result = adjacency_matrix_to_adjacency_list(adj)
        assert set(result.keys()) == {"1", "2", "3"}


class TestFindAncestor:
    """Tests for find_ancestor using a simple chain graph."""

    def _chain_setup(self, n):
        """Build a simple chain 0->1->2->...->n-1."""
        node_names = [str(i) for i in range(n)]
        adj = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            adj[i, i + 1] = 1
        node2idx = {name: idx for idx, name in enumerate(node_names)}
        idx2node = {idx: name for idx, name in enumerate(node_names)}
        return node_names, adj, node2idx, idx2node

    def test_leaf_has_no_ancestors_beyond_itself(self):
        node_names, adj, node2idx, idx2node = self._chain_setup(3)
        node_set = OrderedSet(["0"])
        ancestors = find_ancestor(node_set, node_names, adj, node2idx, idx2node)
        assert "0" in ancestors

    def test_chain_ancestors(self):
        """Node '2' in chain 0->1->2 has ancestors {0, 1, 2}."""
        node_names, adj, node2idx, idx2node = self._chain_setup(3)
        node_set = OrderedSet(["2"])
        ancestors = find_ancestor(node_set, node_names, adj, node2idx, idx2node)
        assert "0" in ancestors
        assert "1" in ancestors
        assert "2" in ancestors


class TestInducedGraph:
    def test_induced_subgraph(self):
        """Inducing on {0,1} from a 3-node chain should give 2x2 submatrix."""
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        node2idx = {"0": 0, "1": 1, "2": 2}
        result = induced_graph({"0", "1"}, adj, node2idx)
        assert result.shape == (2, 2)

    def test_single_node(self):
        adj = np.array([[0, 1], [0, 0]])
        node2idx = {"A": 0, "B": 1}
        result = induced_graph({"A"}, adj, node2idx)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0


class TestGetSimpleOrderedTree:
    def test_chain_structure(self):
        g = get_simple_ordered_tree(4)
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 3
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 2)
        assert g.has_edge(2, 3)

    def test_single_node(self):
        g = get_simple_ordered_tree(1)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0


class TestIsConnected:
    def test_chain_is_connected(self):
        g = get_simple_ordered_tree(5)
        assert is_connected(g) is True

    def test_disconnected_graph(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from([0, 1, 2])
        g.add_edge(0, 1)
        # node 2 is isolated
        assert is_connected(g) is False


class TestConvertToUndirectedGraph:
    def test_edges_preserved(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from([0, 1, 2])
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        u = convert_to_undirected_graph(g)
        assert u.has_edge(0, 1)
        assert u.has_edge(1, 2)

    def test_undirected_result(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_edge(0, 1)
        u = convert_to_undirected_graph(g)
        assert not u.is_directed()


class TestAddAndDelEdge:
    def test_add_edge_creates_edge(self):
        g = get_simple_ordered_tree(3)
        assert not g.has_edge(0, 2)
        add_edge(0, 2, g)
        assert g.has_edge(0, 2)

    def test_add_edge_skipped_if_creates_cycle(self):
        g = get_simple_ordered_tree(3)  # 0->1->2
        add_edge(2, 0, g)  # would create a cycle
        assert not g.has_edge(2, 0)

    def test_del_edge_removes_edge(self):
        g = get_simple_ordered_tree(4)  # 0->1->2->3
        assert g.has_edge(1, 2)
        del_edge(1, 2, g)
        assert not g.has_edge(1, 2)

    def test_del_edge_skipped_if_disconnects(self):
        g = get_simple_ordered_tree(3)  # 0->1->2, removing 0->1 disconnects
        del_edge(0, 1, g)
        # Edge should be restored to keep connectivity
        assert g.has_edge(0, 1)


class TestStrToDot:
    def test_removes_newlines_and_tabs(self):
        raw = "digraph {\n\ta -> b;\n}"
        result = str_to_dot(raw)
        assert "\n" not in result
        assert "\t" not in result

    def test_output_is_string(self):
        raw = "digraph {\n\ta -> b;\n}"
        result = str_to_dot(raw)
        assert isinstance(result, str)


class TestDaggityToDot:
    def test_converts_dag_keyword(self):
        daggity = "dag {\n  x -> y\n}"
        result = daggity_to_dot(daggity)
        assert result.startswith("digraph")

    def test_removes_exposure_attribute(self):
        daggity = "dag { x [exposure] -> y }"
        result = daggity_to_dot(daggity)
        assert "exposure" not in result
