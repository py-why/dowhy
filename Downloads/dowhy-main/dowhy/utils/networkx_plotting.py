from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.drawing import nx_pydot


def plot_causal_graph_networkx(
    causal_graph: nx.Graph,
    layout_prog: Optional[str] = None,
    causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
    colors: Optional[Dict[Union[Any, Tuple[Any, Any]], str]] = None,
    filename: Optional[str] = None,
    display_plot: bool = True,
    label_wrap_length: int = 3,
    figure_size: Optional[Tuple[int, int]] = None,
) -> None:
    if causal_strengths is None:
        causal_strengths = {}
    else:
        causal_strengths = deepcopy(causal_strengths)
    if colors is None:
        colors = {}
    else:
        colors = deepcopy(colors)

    max_strength = 0.0
    for source, target, strength in causal_graph.edges(data="CAUSAL_STRENGTH", default=None):
        if (source, target) not in causal_strengths:
            causal_strengths[(source, target)] = strength

        if causal_strengths[(source, target)] is not None:
            max_strength = max(max_strength, abs(causal_strengths[(source, target)]))

        if (source, target) not in colors:
            colors[(source, target)] = "gray"

    for edge in causal_graph.edges:
        if edge[0] == edge[1]:
            raise ValueError(
                "Node %s has a self-cycle, i.e. a node pointing to itself. Plotting self-cycles is "
                "currently only supported for plots using Graphviz! Consider installing the corresponding "
                "requirements." % edge[0]
            )

    # Wrapping labels if they are too long
    labels = {}
    for node in causal_graph.nodes:
        if node not in colors:
            colors[node] = "skyblue"

        node_name_splits = str(node).split(" ")
        for i in range(1, len(node_name_splits)):
            if len(node_name_splits[i - 1]) > label_wrap_length:
                node_name_splits[i] = "\n" + node_name_splits[i]
            else:
                node_name_splits[i] = " " + node_name_splits[i]

        labels[node] = "".join(node_name_splits)

    from dowhy.utils.plotting import _calc_arrow_width

    edge_widths = {
        (s, t): 2 if causal_strengths[(s, t)] is None else _calc_arrow_width(causal_strengths[(s, t)], max_strength)
        for (s, t) in causal_graph.edges()
    }

    if layout_prog is not None:
        layout = nx_pydot.pydot_layout(causal_graph, prog=layout_prog)
        if figure_size is not None:
            figure = plt.figure(figsize=figure_size)
        else:
            figure = plt.figure()

        nx.draw(
            causal_graph,
            pos=layout,
            node_color=[colors[node] for node in causal_graph.nodes()],
            edge_color=[colors[(s, t)] for (s, t) in causal_graph.edges()],
            labels=labels,
            font_weight="bold",
            node_size=2000,
            arrowsize=20,
            alpha=0.8,
            width=[edge_widths[(s, t)] for (s, t) in causal_graph.edges()],
        )
    else:
        figure = _draw_graph_with_custom_layout(causal_graph, colors, edge_widths, figure_size)

    plt.gca().set_axis_off()
    if display_plot:
        plt.show()

    if filename is not None:
        figure.savefig(filename)


def _draw_graph_with_custom_layout(
    graph: nx.Graph,
    colors: Dict[Any, str],
    edge_widths: Dict[Tuple[Any, Any], float],
    figure_size: Optional[List[int]] = None,
):
    # This layout tries to mimic the graphviz layout in a simpler form. The depth grows horizontally here instead of
    # vertically.
    if isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    layers = _custom_assign_layers(graph)
    nx.set_node_attributes(graph, layers, "layer")
    node_positions = nx.multipartite_layout(graph, subset_key="layer")

    if figure_size is None:
        # Set the figure size based on the number of nodes
        figure = plt.figure(
            figsize=(
                max(np.max([v for v in layers.values()]) * 2, 5),
                max(_custom_count_nodes_vertically(graph) * 1.25, 5),
            )
        )
    else:
        figure = plt.figure(figsize=figure_size)

    nx.draw_networkx_nodes(
        graph,
        node_positions,
        node_size=2000,
        node_color=[colors[node] for node in graph.nodes()],
        alpha=0.8,
    )

    vertical_neighbor_indicator = _custom_create_vertical_neighbor_indicator(graph, node_positions)

    # Nodes that are vertically connected, but not neighbors should be connected via a curved edge.
    edges_with_curved_line = [
        (u, v)
        for u, v in graph.edges()
        if graph.nodes[u]["layer"] == graph.nodes[v]["layer"] and not vertical_neighbor_indicator.loc[str(u), str(v)]
    ]
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=edges_with_curved_line,
        width=[edge_widths[(s, t)] for (s, t) in edges_with_curved_line],
        edge_color=[colors[(s, t)] for (s, t) in edges_with_curved_line],
        arrowsize=20,
        connectionstyle="arc3,rad=0.5",
        alpha=0.8,
        min_source_margin=25,
        min_target_margin=21,
    )

    # All other nodes should be connected with a straight line.
    edges_with_straigth_line = [
        (u, v)
        for u, v in graph.edges()
        if (graph.nodes[u]["layer"] == graph.nodes[v]["layer"] and vertical_neighbor_indicator.loc[str(u), str(v)])
        or graph.nodes[u]["layer"] != graph.nodes[v]["layer"]
    ]
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=edges_with_straigth_line,
        width=[edge_widths[(s, t)] for (s, t) in edges_with_straigth_line],
        edge_color=[colors[(s, t)] for (s, t) in edges_with_straigth_line],
        arrowsize=20,
        alpha=0.8,
        min_source_margin=25,
        min_target_margin=21,
    )

    # Draw labels node labels
    for node, (x, y) in node_positions.items():
        plt.text(x, y, node, ha="center", va="center", color="black", fontweight="bold")

    return figure


def _custom_assign_layers(graph):
    # Each node gets a depth assigned, based on the distance to the closest root node.
    layers = {}

    if not isinstance(graph, nx.DiGraph):
        sub_graphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
        # In case of undirected graphs, we just take any node as root node.
        root_nodes = [list(sub_graph.nodes)[0] for sub_graph in sub_graphs]
    else:
        sub_graphs = [graph]
        root_nodes = [n for n, d in graph.in_degree() if d == 0]

    for sub_graph in sub_graphs:
        nodes_in_subgraph = list(sub_graph.nodes)

        for node in nodes_in_subgraph:
            min_distance = float("inf")

            for root_node in root_nodes:
                try:
                    distance = nx.shortest_path_length(graph, root_node, node)
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    # No path to root node, ignore this connection then.
                    continue

            layers[node] = min_distance

    return layers


def _custom_count_nodes_vertically(graph):
    # Counts the number of vertical nodes in the same layers.
    layer_count = {}
    for n in graph.nodes:
        if graph.nodes[n]["layer"] not in layer_count:
            layer_count[graph.nodes[n]["layer"]] = 0

        layer_count[graph.nodes[n]["layer"]] += 1

    return np.max([v for v in layer_count.items()])


def _custom_create_vertical_neighbor_indicator(graph, pos):
    # Creates a matrix indicating whether two nodes are vertical neighbors.
    all_nodes = list(graph.nodes)
    vertical_neighbor_indicator = pd.DataFrame(
        np.zeros((len(all_nodes), len(all_nodes))).astype(bool),
        index=[str(n) for n in all_nodes],
        columns=[str(n) for n in all_nodes],
    )

    # Get all y coordinates per layer
    layer_y_coords = {}
    for n in graph.nodes:
        if graph.nodes[n]["layer"] not in layer_y_coords:
            layer_y_coords[graph.nodes[n]["layer"]] = []

        layer_y_coords[graph.nodes[n]["layer"]].append((n, pos[n][1]))

    # Sort the y-coordinates
    for layer in layer_y_coords:
        layer_y_coords[layer].sort(key=lambda x: x[1])

    layer_y_coords_map = {}
    for layer in layer_y_coords:
        for i, k in enumerate(layer_y_coords[layer]):
            if k[0] in layer_y_coords_map:
                raise RuntimeError("Something went wrong when creating the layer y-coordinate map.")
            layer_y_coords_map[k[0]] = i

    for n1 in all_nodes:
        for n2 in all_nodes:
            if n1 == n2:
                vertical_neighbor_indicator.loc[str(n1), str(n2)] = True
                continue

            n1_layer = graph.nodes[n1]["layer"]
            n2_layer = graph.nodes[n2]["layer"]

            if n1_layer != n2_layer:
                vertical_neighbor_indicator.loc[str(n1), str(n2)] = False
                continue

            vertical_neighbor_indicator.loc[str(n1), str(n2)] = (
                layer_y_coords_map[n1] == layer_y_coords_map[n2] + 1
            ) or (layer_y_coords_map[n1] == layer_y_coords_map[n2] - 1)

    return vertical_neighbor_indicator
