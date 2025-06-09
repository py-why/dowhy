"""This module defines the fundamental interfaces and functions related to causal graphs."""

import copy
import itertools
import logging
import re
from abc import abstractmethod
from typing import Any, List, Protocol

import networkx as nx
from networkx.algorithms.dag import has_cycle

# version compatibility for breaking change in networkx 3.5
try:
    from networkx.algorithms.d_separation import is_d_separator as d_separated
except ImportError:
    from networkx.algorithms.d_separation import d_separated

from dowhy.utils.api import parse_state
from dowhy.utils.graph_operations import daggity_to_dot

_logger = logging.getLogger(__name__)


class HasNodes(Protocol):
    """This protocol defines a trait for classes having nodes."""

    @property
    @abstractmethod
    def nodes(self):
        """:returns Dict[Any, Dict[Any, Any]]"""
        raise NotImplementedError


class HasEdges(Protocol):
    """This protocol defines a trait for classes having edges."""

    @property
    @abstractmethod
    def edges(self):
        """:returns a Dict[Tuple[Any, Any], Dict[Any, Any]]"""
        raise NotImplementedError


class DirectedGraph(HasNodes, HasEdges, Protocol):
    """A protocol representing a directed graph as needed by graphical causal models.

    This protocol specifically defines a subset of the networkx.DiGraph class, which make that class automatically
    compatible with DirectedGraph. While in most cases a networkx.DiGraph is the class of choice when constructing
    a causal graph, anyone can choose to provide their own implementation of the DirectGraph interface.
    """

    @abstractmethod
    def predecessors(self, node):
        raise NotImplementedError


def is_root_node(causal_graph: DirectedGraph, node: Any) -> bool:
    return list(causal_graph.predecessors(node)) == []


def get_ordered_predecessors(causal_graph: DirectedGraph, node: Any) -> List[Any]:
    """This function returns predecessors of a node in a well-defined order.

    This is necessary, because we select subsets of columns in Dataframes by using a node's parents, and these parents
    might not be returned in a reliable order.
    """
    return sorted(causal_graph.predecessors(node))


def node_connected_subgraph_view(g: DirectedGraph, node: Any) -> Any:
    """Returns a view of the provided graph g that contains only nodes connected to the node passed in"""
    # can't use nx.node_connected_component, because it doesn't work with DiGraphs.
    # Hence, a manual loop:
    return nx.induced_subgraph(g, [n for n in g.nodes if nx.has_path(g, n, node)])


def validate_acyclic(causal_graph: DirectedGraph) -> None:
    if has_cycle(causal_graph):
        raise RuntimeError("The graph contains a cycle, but an acyclic graph is expected!")


def validate_node_in_graph(causal_graph: HasNodes, node: Any) -> None:
    if node not in causal_graph.nodes:
        raise ValueError("Node %s can not be found in the given graph!" % node)


def check_valid_backdoor_set(
    graph: nx.DiGraph,
    nodes1,
    nodes2,
    nodes3,
    backdoor_paths=None,
    new_graph: nx.DiGraph = None,
    dseparation_algo="default",
):
    """Assume that the first parameter (nodes1) is the treatment,
    the second is the outcome, and the third is the candidate backdoor set
    """
    # also return the number of backdoor paths blocked by observed nodes
    if dseparation_algo == "default":
        if new_graph is None:
            # Assume that nodes1 is the treatment
            new_graph = do_surgery(graph, nodes1, remove_outgoing_edges=True)

        dseparated = d_separated(new_graph, set(nodes1), set(nodes2), set(nodes3))
    elif dseparation_algo == "naive":
        # ignores new_graph parameter, always uses self._graph
        if backdoor_paths is None:
            backdoor_paths = get_backdoor_paths(graph, nodes1, nodes2)
        dseparated = all([is_blocked(graph, path, nodes3) for path in backdoor_paths])
    else:
        raise ValueError(f"{dseparation_algo} method for d-separation not supported.")
    return {"is_dseparated": dseparated}


def do_surgery(
    graph: nx.DiGraph,
    node_names,
    remove_outgoing_edges=False,
    remove_incoming_edges=False,
    target_node_names=None,
    remove_only_direct_edges_to_target=False,
):
    """Method to create a new graph based on the concept of do-surgery.

    :param node_names: focal nodes for the surgery
    :param remove_outgoing_edges: whether to remove outgoing edges from the focal nodes
    :param remove_incoming_edges: whether to remove incoming edges to the focal nodes
    :param target_node_names: target nodes (optional) for the surgery, only used when remove_only_direct_edges_to_target is True
    :param remove_only_direct_edges_to_target: whether to remove only the direct edges from focal nodes to the target nodes

    :returns: a new networkx graph after the specified removal of edges
    """

    node_names = parse_state(node_names)
    new_graph = graph.copy()
    for node_name in node_names:
        if remove_outgoing_edges:
            if remove_only_direct_edges_to_target:
                new_graph.remove_edges_from([(node_name, v) for v in target_node_names])
            else:
                children = new_graph.successors(node_name)
                edges_bunch = [(node_name, child) for child in children]
                new_graph.remove_edges_from(edges_bunch)
        if remove_incoming_edges:
            # removal of only direct edges wrt a target is not implemented for incoming edges
            parents = new_graph.predecessors(node_name)
            edges_bunch = [(parent, node_name) for parent in parents]
            new_graph.remove_edges_from(edges_bunch)
    return new_graph


def get_backdoor_paths(graph: nx.DiGraph, nodes1, nodes2):
    paths = []
    undirected_graph = graph.to_undirected()
    nodes12 = set(nodes1).union(nodes2)
    for node1 in nodes1:
        for node2 in nodes2:
            backdoor_paths = [
                pth
                for pth in nx.all_simple_paths(undirected_graph, source=node1, target=node2)
                if graph.has_edge(pth[1], pth[0])
            ]
            # remove paths that have nodes1\node1 or nodes2\node2 as intermediate nodes
            filtered_backdoor_paths = [pth for pth in backdoor_paths if len(nodes12.intersection(pth[1:-1])) == 0]
            paths.extend(filtered_backdoor_paths)
    _logger.debug("Backdoor paths: " + str(paths))
    return paths


def is_blocked(graph: nx.DiGraph, path, conditioned_nodes):
    """Uses d-separation criteria to decide if conditioned_nodes block given path."""

    blocked_by_conditioning = False
    has_unconditioned_collider = False

    for i in range(len(path) - 2):
        if graph.has_edge(path[i], path[i + 1]) and graph.has_edge(path[i + 2], path[i + 1]):  # collider
            collider_descendants = nx.descendants(graph, path[i + 1])
            if path[i + 1] not in conditioned_nodes and all(
                cdesc not in conditioned_nodes for cdesc in collider_descendants
            ):
                has_unconditioned_collider = True
        else:  # chain or fork
            if path[i + 1] in conditioned_nodes:
                blocked_by_conditioning = True
                break
    if blocked_by_conditioning:
        return True
    elif has_unconditioned_collider:
        return True
    else:
        return False


def get_ancestors(graph: nx.DiGraph, nodes):
    ancestors = set()
    for node_name in nodes:
        ancestors = ancestors.union(set(nx.ancestors(graph, node_name)))
    return ancestors


def get_descendants(graph: nx.DiGraph, nodes):
    descendants = set()
    for node_name in nodes:
        descendants = descendants.union(set(nx.descendants(graph, node_name)))
    return descendants


def get_proper_causal_path_nodes(graph: nx.DiGraph, action_nodes, outcome_nodes):
    """Method to get the proper causal path nodes, as described in van der Zander et al. "Constructing Separators and
    Adjustment Sets in Ancestral Graphs", Section 4.1. We cannot use do_surgery() since we require deep copies of the given graph.

    :param graph: the causal graph in question
    :param action_nodes: the action nodes
    :param outcome_nodes: the outcome nodes

    :returns: the set of nodes that lie on proper causal paths from X to Y
    """

    # 1) Create a pair of modified graphs by removing inbound and outbound arrows from the action nodes, respectively.
    graph_post_interv = copy.deepcopy(graph)  # remove incoming arrows to our action nodes
    edges_to_remove = [(u, v) for u, v in graph_post_interv.in_edges(action_nodes)]
    graph_post_interv.remove_edges_from(edges_to_remove)
    graph_with_action_nodes_as_sinks = copy.deepcopy(graph)  # remove outbound arrows from our action nodes
    edges_to_remove = [(u, v) for u, v in graph_with_action_nodes_as_sinks.out_edges(action_nodes)]
    graph_with_action_nodes_as_sinks.remove_edges_from(edges_to_remove)

    # 2) Use the modified graphs to identify the nodes which lie on proper causal paths from the
    # action nodes to the outcome nodes.
    de_x = get_descendants(graph_post_interv, action_nodes).union(action_nodes)
    an_y = get_ancestors(graph_with_action_nodes_as_sinks, outcome_nodes).union(outcome_nodes)
    return (set(de_x) - set(action_nodes)) & an_y


def get_proper_backdoor_graph(graph: nx.DiGraph, action_nodes, outcome_nodes):
    """Method to get the proper backdoor graph from a causal graph, as described in van der Zander et al. "Constructing Separators and
    Adjustment Sets in Ancestral Graphs", Section 4.1. We cannot use do_surgery() since we require deep copies of the given graph.

    :param graph: the causal graph in question
    :param action_nodes: the action nodes
    :param outcome_nodes: the outcome nodes

    :returns: a new graph which is the proper backdoor graph of the original
    """

    # First we can just call get_proper_causal_path_nodes, then
    # we remove edges from the action_nodes to the proper causal path nodes.
    graph_pbd = copy.deepcopy(graph)
    graph_pbd.remove_edges_from(
        [(u, v) for u in action_nodes for v in get_proper_causal_path_nodes(graph, action_nodes, outcome_nodes)]
    )
    return graph_pbd


def check_dseparation(graph: nx.DiGraph, nodes1, nodes2, nodes3, new_graph=None, dseparation_algo="default"):
    if dseparation_algo == "default":
        if new_graph is None:
            new_graph = graph
        dseparated = d_separated(new_graph, set(nodes1), set(nodes2), set(nodes3))
    else:
        raise ValueError(f"{dseparation_algo} method for d-separation not supported.")
    return dseparated


def get_all_nodes(graph: nx.DiGraph, observed_nodes: List[Any], include_unobserved_nodes: bool) -> List[Any]:
    observed_nodes = set(observed_nodes)
    return [node for node in graph.nodes if include_unobserved_nodes or node in observed_nodes]


def get_instruments(graph: nx.DiGraph, treatment_nodes, outcome_nodes):
    treatment_nodes = parse_state(treatment_nodes)
    outcome_nodes = parse_state(outcome_nodes)
    parents_treatment = set()
    for node in treatment_nodes:
        parents_treatment = parents_treatment.union(set(graph.predecessors(node)))
    g_no_parents_treatment = do_surgery(graph, treatment_nodes, remove_incoming_edges=True)
    ancestors_outcome = set()
    for node in outcome_nodes:
        ancestors_outcome = ancestors_outcome.union(nx.ancestors(g_no_parents_treatment, node))
    # [TODO: double check these work with multivariate implementation:]
    # Exclusion
    candidate_instruments = parents_treatment.difference(ancestors_outcome)
    _logger.debug("Candidate instruments after satisfying exclusion: %s", candidate_instruments)
    # As-if-random setup
    children_causes_outcome = [nx.descendants(g_no_parents_treatment, v) for v in ancestors_outcome]
    children_causes_outcome = set([item for sublist in children_causes_outcome for item in sublist])

    # As-if-random
    instruments = candidate_instruments.difference(children_causes_outcome)
    _logger.debug("Candidate instruments after satisfying exclusion and as-if-random: %s", instruments)
    return list(instruments)


def check_valid_frontdoor_set(
    graph: nx.DiGraph,
    nodes1,
    nodes2,
    candidate_nodes,
    frontdoor_paths=None,
    new_graph: nx.DiGraph = None,
    dseparation_algo="default",
):
    """Check if valid the frontdoor variables for set of treatments, nodes1 to set of outcomes, nodes2."""
    # Condition 1: node 1 ---> node 2 is intercepted by candidate_nodes
    if dseparation_algo == "default":
        if new_graph is None:
            new_graph = graph
        dseparated = d_separated(new_graph, set(nodes1), set(nodes2), set(candidate_nodes))
    elif dseparation_algo == "naive":
        if frontdoor_paths is None:
            frontdoor_paths = get_all_directed_paths(graph, nodes1, nodes2)

        dseparated = all([is_blocked(graph, path, candidate_nodes) for path in frontdoor_paths])
    else:
        raise ValueError(f"{dseparation_algo} method for d-separation not supported.")
    return dseparated


def get_all_directed_paths(graph: nx.DiGraph, nodes1, nodes2):
    """Get all directed paths between sets of nodes.

    Currently only supports singleton sets.
    """
    if len(nodes1) > 1 or len(nodes2) > 1:
        raise ValueError(
            "The list of action and outcome nodes can only contain one element, i.e., needs to be univariate!"
        )
    return [p for p in nx.all_simple_paths(graph, source=nodes1[0], target=nodes2[0])]


def has_directed_path(graph: nx.DiGraph, action_nodes, outcome_nodes):
    """Checks if there is any directed path between two sets of nodes.

    Returns True if and only if every one of the treatments has at least one direct
    path to one of the outcomes. And, every one of the outcomes has a direct path from
    at least one of the treatments.
    """
    outcome_node_candidates = set()
    action_node_candidates = set()
    for node in action_nodes:
        outcome_node_candidates.update(nx.descendants(graph, node))
    for node in outcome_nodes:
        action_node_candidates.update(nx.ancestors(graph, node))
    return set(outcome_nodes).issubset(outcome_node_candidates) and set(action_nodes).issubset(action_node_candidates)


def check_valid_mediation_set(graph: nx.DiGraph, nodes1, nodes2, candidate_nodes, mediation_paths=None):
    """Check if candidate nodes are valid mediators for set of treatments, nodes1 to set of outcomes, nodes2."""
    if mediation_paths is None:
        mediation_paths = get_all_directed_paths(graph, nodes1, nodes2)

    is_mediator = any([is_blocked(graph, path, candidate_nodes) for path in mediation_paths])
    return is_mediator


def get_adjacency_matrix(graph: nx.DiGraph, *args, **kwargs):
    """
    Get adjacency matrix from the networkx graph

    """
    return nx.convert_matrix.to_numpy_array(graph, *args, **kwargs)


def build_graph(
    action_nodes: List[str],
    outcome_nodes: List[str],
    common_cause_nodes: List[str] = None,
    instrument_nodes=None,
    effect_modifier_nodes=None,
    mediator_nodes=None,
):
    """Creates nodes and edges based on variable names and their semantics.

    Currently only considers the graphical representation of "direct" effect modifiers. Thus, all effect modifiers are assumed to be "direct" unless otherwise expressed using a graph. Based on the taxonomy of effect modifiers by VanderWheele and Robins: "Four types of effect modification: A classification based on directed acyclic graphs. Epidemiology. 2007."
    """
    graph = nx.DiGraph()

    action_nodes = parse_state(action_nodes)
    outcome_nodes = parse_state(outcome_nodes)
    common_cause_nodes = parse_state(common_cause_nodes)
    instrument_nodes = parse_state(instrument_nodes)
    effect_modifier_nodes = parse_state(effect_modifier_nodes)

    for treatment in action_nodes:
        graph.add_node(treatment)
    for outcome in outcome_nodes:
        graph.add_node(outcome)
    for treatment, outcome in itertools.product(action_nodes, outcome_nodes):
        graph.add_edge(treatment, outcome)

    # Adding common causes
    if common_cause_nodes:
        for node_name in common_cause_nodes:
            for treatment, outcome in itertools.product(action_nodes, outcome_nodes):
                graph.add_node(node_name)
                graph.add_edge(node_name, treatment)
                graph.add_edge(node_name, outcome)

    # Adding instruments
    if instrument_nodes:
        if type(instrument_nodes[0]) != tuple:
            if len(action_nodes) > 1:
                _logger.info("Assuming Instrument points to all treatments! Use tuples for more granularity.")
            for instrument, treatment in itertools.product(instrument_nodes, action_nodes):
                graph.add_node(instrument)
                graph.add_edge(instrument, treatment)
        else:
            for instrument, treatment in itertools.product(instrument_nodes):
                graph.add_node(instrument)
                graph.add_edge(instrument, treatment)

    # Adding effect modifiers
    if effect_modifier_nodes:
        for node_name in effect_modifier_nodes:
            if node_name not in common_cause_nodes:
                for outcome in outcome_nodes:
                    graph.add_node(node_name)
                    # Assuming the simple form of effect modifier
                    # that directly causes the outcome.
                    graph.add_edge(node_name, outcome)
                    # self._graph.add_edge(node_name, outcome, style = "dotted", headport="s", tailport="n")
                    # self._graph.add_edge(outcome, node_name, style = "dotted", headport="n", tailport="s") # TODO make the ports more general so that they apply not just to top-bottom node configurations
    if mediator_nodes:
        for node_name in mediator_nodes:
            for treatment, outcome in itertools.product(action_nodes, outcome_nodes):
                graph.add_node(node_name)
                graph.add_edge(treatment, node_name)
                graph.add_edge(node_name, outcome)

    return graph


def build_graph_from_str(graph_str: str) -> nx.DiGraph:
    """
    User-friendly function that returns a networkx graph based on the graph string.

    Formats supported: dot, gml, daggity

    The `graph_str` parameter can refer to the path of a text file containing the encoded graph or contain the actual encoded graph as a string.

    :param graph_str: a string containing the filepath or the encoded graph
    :type graph_str: str

    :returns: a networkx directed graph object
    """
    # some preprocessing steps
    if re.match(r".*\.txt", graph_str):
        text_file = open(graph_str, "r")
        graph_str = text_file.read()
        text_file.close()
    if re.match(r"^dag", graph_str):  # Convert daggity output to dot format
        graph_str = daggity_to_dot(graph_str)
    if isinstance(graph_str, str):
        graph_str = graph_str.replace("\n", " ")

    # parsing the correct graph based on input graph format
    if re.match(r".*\.dot", graph_str):
        # load dot file
        try:
            import pygraphviz as pgv

            return nx.DiGraph(nx.drawing.nx_agraph.read_dot(graph_str))
        except Exception as e:
            _logger.error("Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot...")
            try:
                import pydot

                return nx.DiGraph(nx.drawing.nx_pydot.read_dot(graph_str))
            except Exception as e:
                _logger.error("Error: Pydot cannot be loaded. " + str(e))
                raise e
    elif re.match(r".*\.gml", graph_str):
        return nx.DiGraph(nx.read_gml(graph_str))
    elif re.match(r".*graph\s*\{.*\}\s*", graph_str):
        try:
            import pygraphviz as pgv

            graph = pgv.AGraph(graph_str, strict=True, directed=True)
            return nx.drawing.nx_agraph.from_agraph(graph)
        except Exception as e:
            _logger.error("Error: Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot ...")
            try:
                import pydot

                P_list = pydot.graph_from_dot_data(graph_str)
                return nx.DiGraph(nx.drawing.nx_pydot.from_pydot(P_list[0]))
            except Exception as e:
                _logger.error("Error: Pydot cannot be loaded. " + str(e))
                raise e
    elif re.match(".*graph\s*\[.*\]\s*", graph_str):
        return nx.DiGraph(nx.parse_gml(graph_str))
    else:
        _logger.error("Error: Please provide graph (as string or text file) in dot or gml format.")
        _logger.error("Error: Incorrect graph format")
        raise ValueError
