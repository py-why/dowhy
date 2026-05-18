"""
Bridge identifier for z-Identifiability via surrogate experiments.

Implements ZIDIdentifier, which integrates the complete z-ID decision procedure
(Bareinboim & Pearl, 2012) into DoWhy's identification pipeline by translating
DoWhy's graph representation into ananke-causal's ADMG format.
"""

import itertools
import logging
from typing import List, Tuple

import networkx as nx
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand

logger = logging.getLogger(__name__)


class ZIDIdentifier:
    """
    Bridge identifier for complete z-Identifiability using surrogate
    experiments, powered by ananke-causal.

    DoWhy represents latent confounders as unobserved nodes (observed='no')
    with directed edges to their children. This class converts that
    representation into ananke's bidirected-edge ADMG format before running
    the z-ID decision procedure.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        action_nodes: List[str],
        outcome_nodes: List[str],
        surrogate_nodes: List[str],
    ):
        self.graph = graph
        self.action_nodes = list(action_nodes)
        self.outcome_nodes = list(outcome_nodes)
        self.surrogate_nodes = list(surrogate_nodes)

    def identify_effect(self) -> IdentifiedEstimand:
        """
        Run the z-ID decision procedure and return an IdentifiedEstimand.
        Identifying functional on success: P(y|do(x)) = sum_z P(y|x,z) P(z).
        Raises Exception if not z-identifiable.
        """
        try:
            from ananke.graphs import ADMG as _ADMG
            from ananke.identification.idz import idz_id as _idz_id
        except ImportError:
            raise ImportError(
                "ananke-causal is required for ZIDIdentifier. "
                "Install it with: pip install ananke-causal"
            )

        ananke_graph = self._convert_to_ananke(self.graph)

        logger.debug(
            "ZIDIdentifier: vertices=%s  di_edges=%s  bi_edges=%s",
            ananke_graph.vertices, ananke_graph.di_edges, ananke_graph.bi_edges,
        )

        is_identifiable = _idz_id(
            graph=ananke_graph,
            treatments=self.action_nodes,
            outcomes=self.outcome_nodes,
            surrogates=self.surrogate_nodes,
        )

        if not is_identifiable:
            raise Exception(
                f"P({self.outcome_nodes} | do({self.action_nodes})) is NOT "
                f"z-identifiable given surrogates Z={self.surrogate_nodes}."
            )

        logger.debug("ZIDIdentifier: effect IS z-identifiable. Functional: sum_z P(y|x,z)P(z)")

        return IdentifiedEstimand(
            None,
            treatment_variable=self.action_nodes,
            outcome_variable=self.outcome_nodes,
            estimand_type="nonparametric-ate",
            estimands={"backdoor": None},
            backdoor_variables=self.surrogate_nodes,
            instrumental_variables=None,
            frontdoor_variables=None,
            mediation_first_stage_confounders=None,
            mediation_second_stage_confounders=None,
        )

    def _convert_to_ananke(self, nx_graph: nx.DiGraph) -> "ADMG":
        """
        Convert DoWhy nx.DiGraph → ananke ADMG.

        Strategy 1 (primary): nodes with observed='no' are latent confounders.
          Each such node U with observed children {C1,...,Cn} → C(n,2) bi_edges.
          U and its di_edges are excluded from the ADMG.

        Strategy 2 (fallback): edges with style='bidirected', bidirected=True,
          or arrowhead='both' are collected as bi_edges directly.
        """
        from ananke.graphs import ADMG

        di_edges: List[Tuple[str, str]] = []
        bi_edges: List[Tuple[str, str]] = []
        latent_nodes: set = set()

        # Strategy 1
        for node, attrs in nx_graph.nodes(data=True):
            if attrs.get("observed", "yes") == "no":
                latent_nodes.add(node)
                children = list(nx_graph.successors(node))
                obs_children = [
                    c for c in children
                    if nx_graph.nodes[c].get("observed", "yes") == "yes"
                ]
                if len(obs_children) >= 2:
                    for a, b in itertools.combinations(obs_children, 2):
                        bi_edges.append((a, b))
                elif len(obs_children) == 1:
                    logger.warning("Latent node '%s' has only 1 observed child; no bi_edge emitted.", node)
                else:
                    logger.warning("Latent node '%s' has no observed children; skipped.", node)

        # Strategy 2 + directed edges
        for u, v, attrs in nx_graph.edges(data=True):
            if (attrs.get("style") == "bidirected"
                    or attrs.get("bidirected") is True
                    or attrs.get("arrowhead") == "both"):
                bi_edges.append((u, v))
            else:
                if u not in latent_nodes and v not in latent_nodes:
                    di_edges.append((u, v))

        vertices = [
            n for n in nx_graph.nodes()
            if nx_graph.nodes[n].get("observed", "yes") == "yes"
        ]

        # Deduplicate bidirected edges (unordered pairs)
        seen: set = set()
        unique_bi: List[Tuple[str, str]] = []
        for a, b in bi_edges:
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                unique_bi.append((a, b))

        return ADMG(vertices=vertices, di_edges=di_edges, bi_edges=unique_bi)