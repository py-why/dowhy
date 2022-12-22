import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pygraphviz
from matplotlib import image, pyplot


def _plot_causal_graph_graphviz(
    causal_graph: nx.Graph,
    display_causal_strengths: bool = True,
    causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
    colors: Optional[Dict[Union[Any, Tuple[Any, Any]], str]] = None,
    filename: Optional[str] = None,
    display_plot: bool = True,
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
    for (source, target, strength) in causal_graph.edges(data="CAUSAL_STRENGTH", default=None):
        if (source, target) not in causal_strengths:
            causal_strengths[(source, target)] = strength
        if causal_strengths[(source, target)] is not None:
            max_strength = max(max_strength, abs(causal_strengths[(source, target)]))
        if (source, target) not in colors:
            colors[(source, target)] = "black"

    pygraphviz_graph = pygraphviz.AGraph(directed=isinstance(causal_graph, nx.DiGraph))

    for node in causal_graph.nodes:
        if node in colors:
            pygraphviz_graph.add_node(node, color=colors[node], fontcolor=colors[node])
        else:
            pygraphviz_graph.add_node(node)

    for (source, target) in causal_graph.edges():
        causal_strength = causal_strengths[(source, target)]
        color = colors[(source, target)]
        if causal_strength is not None:
            if np.isinf(causal_strength):
                causal_strength = 10000
                tmp_label = "Inf"
            else:
                tmp_label = str(" %s" % str(int(causal_strength * 100) / 100))

            pygraphviz_graph.add_edge(
                str(source),
                str(target),
                label=tmp_label if display_causal_strengths else None,
                penwidth=str(_calc_arrow_width(causal_strength, max_strength)),
                color=color,
            )
        else:
            pygraphviz_graph.add_edge(str(source), str(target), color=color)

    pygraphviz_graph.layout(prog="dot")
    if filename is not None:
        filename, file_extension = os.path.splitext(filename)
        if file_extension == "":
            file_extension = ".pdf"
        pygraphviz_graph.draw(filename + file_extension)

    if display_plot:
        _plot_as_pyplot_figure(pygraphviz_graph, figure_size)


def _calc_arrow_width(strength: float, max_strength: float):
    return 0.1 + 4.0 * float(abs(strength)) / float(max_strength)


def _plot_as_pyplot_figure(pygraphviz_graph: pygraphviz.AGraph, figure_size: Optional[Tuple[int, int]] = None) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pygraphviz_graph.draw(tmp_dir_name + os.sep + "Graph.png")
        img = image.imread(tmp_dir_name + os.sep + "Graph.png")

        if figure_size is not None:
            org_fig_size = pyplot.rcParams["figure.figsize"]
            pyplot.rcParams["figure.figsize"] = figure_size

        pyplot.imshow(img)
        pyplot.axis("off")
        pyplot.show()

        if figure_size is not None:
            pyplot.rcParams["figure.figsize"] = org_fig_size
