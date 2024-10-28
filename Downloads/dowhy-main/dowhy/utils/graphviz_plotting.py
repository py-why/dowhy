import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pygraphviz


def plot_causal_graph_graphviz(
    causal_graph: nx.Graph,
    layout_prog: Optional[str] = None,
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

    if layout_prog is None:
        layout_prog = "dot"

    max_strength = 0.0
    for source, target, strength in causal_graph.edges(data="CAUSAL_STRENGTH", default=None):
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

    for source, target in causal_graph.edges():
        causal_strength = causal_strengths[(source, target)]
        color = colors[(source, target)]
        if causal_strength is not None:
            if np.isinf(causal_strength):
                causal_strength = 10000
                tmp_label = "Inf"
            else:
                tmp_label = str(" %s" % str(int(causal_strength * 100) / 100))

            from dowhy.utils.plotting import _calc_arrow_width

            pygraphviz_graph.add_edge(
                str(source),
                str(target),
                label=tmp_label if display_causal_strengths else None,
                penwidth=str(_calc_arrow_width(causal_strength, max_strength)),
                color=color,
            )
        else:
            pygraphviz_graph.add_edge(str(source), str(target), color=color)

    pygraphviz_graph.layout(prog=layout_prog)
    if filename is not None:
        filename, file_extension = os.path.splitext(filename)
        if file_extension == "":
            file_extension = ".pdf"
        pygraphviz_graph.draw(filename + file_extension)

    if display_plot:
        from dowhy.utils.plotting import _plot_as_pyplot_figure

        _plot_as_pyplot_figure(pygraphviz_graph, figure_size)
