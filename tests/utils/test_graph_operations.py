"""
Unit tests for dowhy.utils.graph_operations.

These functions are core utilities used by the ID algorithm
(id_identifier.py) for causal identification. They were previously
exercised only indirectly through higher-level integration tests.
"""

import networkx as nx
import numpy as np

from dowhy.utils.graph_operations import (
    add_edge,
    adjacency_matrix_to_adjacency_list,
    daggity_to_dot,
    del_edge,
    find_ancestor,
    find_c_components,
    get_simple_ordered_tree,
    induced_graph,
    is_connected,
    str_to_dot,
)
from dowhy.utils.ordered_set import OrderedSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mapping(node_names):
    """Return (node2idx, idx2node) dicts for a list of node names."""
    node2idx = {n: i for i, n in enumerate(node_names)}
    idx2node = {i: n for i, n in enumerate(node_names)}
    return node2idx, idx2node


# ---------------------------------------------------------------------------
# adjacency_matrix_to_adjacency_list
# ---------------------------------------------------------------------------


class TestAdjacencyMatrixToAdjacencyList:
    def test_chain_graph(self):
        # A → B → C
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        result = adjacency_matrix_to_adjacency_list(adj, ["A", "B", "C"])
        assert result == {"A": ["B"], "B": ["C"], "C": []}

    def test_default_integer_labels(self):
        adj = np.array([[0, 1], [0, 0]])
        result = adjacency_matrix_to_adjacency_list(adj)
        assert result == {"1": ["2"], "2": []}

    def test_no_edges(self):
        adj = np.zeros((3, 3), dtype=int)
        result = adjacency_matrix_to_adjacency_list(adj, ["X", "Y", "Z"])
        assert result == {"X": [], "Y": [], "Z": []}

    def test_fully_connected_dag(self):
        # A → B, A → C, B → C
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        result = adjacency_matrix_to_adjacency_list(adj, ["A", "B", "C"])
        assert set(result["A"]) == {"B", "C"}
        assert result["B"] == ["C"]
        assert result["C"] == []


# ---------------------------------------------------------------------------
# find_ancestor
# ---------------------------------------------------------------------------


class TestFindAncestor:
    def _chain_fixtures(self):
        """Return fixtures for A → B → C graph."""
        names = ["A", "B", "C"]
        node2idx, idx2node = _make_mapping(names)
        adj = np.matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # A→B→C
        node_names = OrderedSet(names)
        return adj, node_names, node2idx, idx2node

    def test_chain_leaf_has_all_ancestors(self):
        adj, node_names, node2idx, idx2node = self._chain_fixtures()
        ancestors = find_ancestor(OrderedSet(["C"]), node_names, adj, node2idx, idx2node)
        assert set(ancestors.get_all()) == {"A", "B", "C"}

    def test_chain_root_only_has_itself(self):
        adj, node_names, node2idx, idx2node = self._chain_fixtures()
        ancestors = find_ancestor(OrderedSet(["A"]), node_names, adj, node2idx, idx2node)
        assert set(ancestors.get_all()) == {"A"}

    def test_chain_middle_node(self):
        adj, node_names, node2idx, idx2node = self._chain_fixtures()
        ancestors = find_ancestor(OrderedSet(["B"]), node_names, adj, node2idx, idx2node)
        assert set(ancestors.get_all()) == {"A", "B"}

    def test_diamond_graph(self):
        # A → B, A → C, B → D, C → D
        names = ["A", "B", "C", "D"]
        node2idx, idx2node = _make_mapping(names)
        adj = np.matrix(
            [
                [0, 1, 1, 0],  # A → B, A → C
                [0, 0, 0, 1],  # B → D
                [0, 0, 0, 1],  # C → D
                [0, 0, 0, 0],
            ]
        )
        node_names = OrderedSet(names)
        ancestors = find_ancestor(OrderedSet(["D"]), node_names, adj, node2idx, idx2node)
        assert set(ancestors.get_all()) == {"A", "B", "C", "D"}

    def test_multiple_query_nodes(self):
        adj, node_names, node2idx, idx2node = self._chain_fixtures()
        # Ancestors of {B, C} should be all nodes
        ancestors = find_ancestor(OrderedSet(["B", "C"]), node_names, adj, node2idx, idx2node)
        assert set(ancestors.get_all()) == {"A", "B", "C"}

    def test_isolated_node(self):
        names = ["X", "Y", "Z"]
        node2idx, idx2node = _make_mapping(names)
        adj = np.matrix(np.zeros((3, 3), dtype=int))
        node_names = OrderedSet(names)
        ancestors = find_ancestor(OrderedSet(["Y"]), node_names, adj, node2idx, idx2node)
        # No edges, so only Y itself
        assert set(ancestors.get_all()) == {"Y"}


# ---------------------------------------------------------------------------
# induced_graph
# ---------------------------------------------------------------------------


class TestInducedGraph:
    def test_chain_subgraph(self):
        # A → B → C; induced on {A, B} should give A → B only
        names = ["A", "B", "C"]
        node2idx, _ = _make_mapping(names)
        adj = np.matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        subset = OrderedSet(["A", "B"])
        induced = induced_graph(subset, adj, node2idx)
        assert induced.shape == (2, 2)
        # A→B edge preserved
        assert induced[0, 1] == 1
        # No other edges
        assert induced[0, 0] == 0
        assert induced[1, 0] == 0
        assert induced[1, 1] == 0

    def test_single_node(self):
        names = ["A", "B"]
        node2idx, _ = _make_mapping(names)
        adj = np.matrix([[0, 1], [0, 0]])
        subset = OrderedSet(["A"])
        induced = induced_graph(subset, adj, node2idx)
        assert induced.shape == (1, 1)
        assert induced[0, 0] == 0


# ---------------------------------------------------------------------------
# find_c_components
# ---------------------------------------------------------------------------


class TestFindCComponents:
    def test_no_bidirected_edges(self):
        # A → B → C: all directed, so each node is its own c-component
        names = ["A", "B", "C"]
        _, idx2node = _make_mapping(names)
        adj = np.matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        node_names = OrderedSet(names)
        components = find_c_components(adj, node_names, idx2node)
        assert len(components) == 3
        sizes = sorted(len(list(c.get_all())) for c in components)
        assert sizes == [1, 1, 1]

    def test_single_bidirected_edge(self):
        # A ↔ B (bidirected), C isolated
        names = ["A", "B", "C"]
        _, idx2node = _make_mapping(names)
        adj = np.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        node_names = OrderedSet(names)
        components = find_c_components(adj, node_names, idx2node)
        assert len(components) == 2
        component_sizes = sorted(len(list(c.get_all())) for c in components)
        assert component_sizes == [1, 2]
        # The bigger component should contain both A and B
        big = next(c for c in components if len(list(c.get_all())) == 2)
        assert set(big.get_all()) == {"A", "B"}

    def test_all_bidirected(self):
        # A ↔ B ↔ C (all connected via bidirected)
        names = ["A", "B", "C"]
        _, idx2node = _make_mapping(names)
        adj = np.matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        node_names = OrderedSet(names)
        components = find_c_components(adj, node_names, idx2node)
        assert len(components) == 1
        assert set(components[0].get_all()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# get_simple_ordered_tree
# ---------------------------------------------------------------------------


class TestGetSimpleOrderedTree:
    def test_structure(self):
        g = get_simple_ordered_tree(4)
        assert list(g.nodes()) == [0, 1, 2, 3]
        assert list(g.edges()) == [(0, 1), (1, 2), (2, 3)]

    def test_single_node(self):
        g = get_simple_ordered_tree(1)
        assert list(g.nodes()) == [0]
        assert list(g.edges()) == []

    def test_two_nodes(self):
        g = get_simple_ordered_tree(2)
        assert list(g.edges()) == [(0, 1)]


# ---------------------------------------------------------------------------
# add_edge / del_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_adds_valid_edge(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        add_edge(0, 2, g)  # 0 → 2 does not create a cycle
        assert g.has_edge(0, 2)

    def test_rejects_cycle_forming_edge(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        add_edge(2, 0, g)  # 2 → 0 would close 0→1→2→0
        assert not g.has_edge(2, 0)


class TestDelEdge:
    def test_removes_edge_when_graph_stays_connected(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)  # extra path keeps connectivity
        del_edge(0, 2, g)
        assert not g.has_edge(0, 2)
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 2)

    def test_keeps_edge_when_removal_disconnects(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        del_edge(1, 2, g)  # removing 1→2 disconnects node 2
        assert g.has_edge(1, 2)

    def test_noop_on_missing_edge(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        edges_before = set(g.edges())
        del_edge(0, 99, g)  # edge does not exist
        assert set(g.edges()) == edges_before


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


class TestIsConnected:
    def test_connected_chain(self):
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        assert is_connected(g)

    def test_disconnected_graph(self):
        g = nx.DiGraph()
        g.add_node(0)
        g.add_node(1)
        # No edges between them
        assert not is_connected(g)


# ---------------------------------------------------------------------------
# daggity_to_dot
# ---------------------------------------------------------------------------


class TestDaggityToDot:
    def test_removes_exposure_label(self):
        daggity = "dag { X [exposure]; Y [outcome]; X -> Y }"
        result = daggity_to_dot(daggity)
        assert "exposure" not in result
        assert "digraph" in result

    def test_removes_adjusted_label(self):
        daggity = "dag { Z [adjusted]; X -> Z; Z -> Y }"
        result = daggity_to_dot(daggity)
        assert "adjusted" not in result

    def test_removes_latent_label(self):
        daggity = "dag { U [latent]; U -> X; U -> Y }"
        result = daggity_to_dot(daggity)
        assert "latent" not in result
        assert 'observed="no"' in result

    def test_preserves_edges(self):
        daggity = "dag { X [exposure]; Y [outcome]; X -> Y }"
        result = daggity_to_dot(daggity)
        assert "X -> Y" in result


# ---------------------------------------------------------------------------
# str_to_dot
# ---------------------------------------------------------------------------


class TestStrToDot:
    def test_converts_graphviz_output(self):
        # Simulate the kind of string graphviz library produces
        graphviz_str = "digraph {\n\tA -> B\n}"
        result = str_to_dot(graphviz_str)
        # Result should be a single-line DOT string
        assert "\n" not in result
        assert "\t" not in result
