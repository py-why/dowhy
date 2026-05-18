"""
Tests for ZIDIdentifier and z-ID integration in identify_effect_auto.

Covers:
  - ADMG conversion (unobserved nodes, explicit bidirected attrs)
  - identify_effect() decision procedure (True/False cases)
  - Integration via identify_effect_auto with surrogate_nodes parameter
"""

import sys

import networkx as nx
import pytest

sys.path.insert(
    0,
    "/Users/omarcamara/Desktop/Spring 2026/Research/causal-code/ananke-dev",
)

from dowhy.causal_identifier.auto_identifier import EstimandType, identify_effect_auto
from dowhy.causal_identifier.zid_identifier import ZIDIdentifier


def build_graph(di_edges, confounders=None):
    """Build a DoWhy-style nx.DiGraph with unobserved common-cause nodes."""
    G = nx.DiGraph()
    observed = set()
    for u, v in di_edges:
        G.add_edge(u, v)
        observed.update([u, v])
    for n in observed:
        G.nodes[n]["observed"] = "yes"
    for i, (a, b) in enumerate(confounders or []):
        uid = f"__U_{i}__"
        G.add_node(uid, observed="no")
        G.add_edge(uid, a)
        G.add_edge(uid, b)
    return G


class TestZIDIdentifierConversion:
    def test_no_confounders(self):
        G = build_graph([("X", "Y")])
        zid = ZIDIdentifier(G, ["X"], ["Y"], [])
        admg = zid._convert_to_ananke(G)
        assert set(admg.vertices) == {"X", "Y"}
        assert set(map(tuple, admg.di_edges)) == {("X", "Y")}
        assert list(admg.bi_edges) == []

    def test_single_latent_confounder(self):
        G = build_graph([("X", "Y")], confounders=[("X", "Y")])
        zid = ZIDIdentifier(G, ["X"], ["Y"], [])
        admg = zid._convert_to_ananke(G)
        assert set(admg.vertices) == {"X", "Y"}
        assert {frozenset(e) for e in admg.bi_edges} == {frozenset(("X", "Y"))}

    def test_multi_child_latent_emits_all_pairs(self):
        G = nx.DiGraph()
        for n in ["A", "B", "C"]:
            G.add_node(n, observed="yes")
        G.add_node("U", observed="no")
        for c in ["A", "B", "C"]:
            G.add_edge("U", c)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        zid = ZIDIdentifier(G, ["A"], ["C"], ["B"])
        admg = zid._convert_to_ananke(G)
        bi = {frozenset(e) for e in admg.bi_edges}
        assert bi == {frozenset(("A", "B")), frozenset(("A", "C")), frozenset(("B", "C"))}

    def test_explicit_bidirected_attr(self):
        G = nx.DiGraph()
        G.add_node("X", observed="yes")
        G.add_node("Y", observed="yes")
        G.add_edge("X", "Y")
        G.add_edge("Y", "X", style="bidirected")
        zid = ZIDIdentifier(G, ["X"], ["Y"], [])
        admg = zid._convert_to_ananke(G)
        assert frozenset(("X", "Y")) in {frozenset(e) for e in admg.bi_edges}


class TestZIDIdentifierDecision:
    def test_not_identifiable_no_surrogates(self):
        G = build_graph([("X", "Y")], confounders=[("X", "Y")])
        zid = ZIDIdentifier(G, ["X"], ["Y"], [])
        with pytest.raises(Exception, match="NOT z-identifiable"):
            zid.identify_effect()

    def test_no_confounders_identifiable(self):
        G = build_graph([("X", "Y")])
        zid = ZIDIdentifier(G, ["X"], ["Y"], [])
        estimand = zid.identify_effect()
        assert estimand.backdoor_variables == []

    def test_rescue_case(self):
        """Z->X->Y, W1<->X, W1<->Y, W2<->Z, X<->Z, Y<->Z: z-ID succeeds."""
        G = build_graph(
            [("W_1", "Z"), ("X", "Y"), ("Z", "X")],
            confounders=[("W_1", "X"), ("W_1", "Y"), ("W_2", "Z"), ("X", "Z"), ("Y", "Z")],
        )
        zid = ZIDIdentifier(G, ["X"], ["Y"], ["Z"])
        estimand = zid.identify_effect()
        assert estimand.backdoor_variables == ["Z"]


class TestZIDAutoIdentifier:
    def test_surrogate_nodes_populates_zid_key(self):
        G = build_graph(
            [("W_1", "Z"), ("X", "Y"), ("Z", "X")],
            confounders=[("W_1", "X"), ("W_1", "Y"), ("W_2", "Z"), ("X", "Z"), ("Y", "Z")],
        )
        observed = [n for n in G.nodes if G.nodes[n].get("observed", "yes") == "yes"]
        estimand = identify_effect_auto(
            G,
            action_nodes=["X"],
            outcome_nodes=["Y"],
            observed_nodes=observed,
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            surrogate_nodes=["Z"],
        )
        assert "zid" in estimand.estimands
        assert estimand.estimands["zid"] is not None

    def test_no_surrogates_zid_key_is_none(self):
        G = build_graph([("X", "Y")])
        estimand = identify_effect_auto(
            G,
            action_nodes=["X"],
            outcome_nodes=["Y"],
            observed_nodes=["X", "Y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        assert estimand.estimands.get("zid") is None

    def test_backdoor_none_zid_rescues(self):
        """Pure rescue: backdoor=None but zid succeeds."""
        G = build_graph(
            [("W_1", "Z"), ("X", "Y"), ("Z", "X")],
            confounders=[("W_1", "X"), ("W_1", "Y"), ("W_2", "Z"), ("X", "Z"), ("Y", "Z")],
        )
        observed = [n for n in G.nodes if G.nodes[n].get("observed", "yes") == "yes"]
        estimand = identify_effect_auto(
            G,
            action_nodes=["X"],
            outcome_nodes=["Y"],
            observed_nodes=observed,
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            surrogate_nodes=["Z"],
        )
        assert estimand.estimands["backdoor"] is None
        assert estimand.estimands["zid"] is not None
