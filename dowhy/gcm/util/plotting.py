import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot
from networkx.drawing import nx_pydot

_logger = logging.getLogger(__name__)


def plot(
    causal_graph: nx.Graph,
    causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
    colors: Optional[Dict[Union[Any, Tuple[Any, Any]], str]] = None,
    filename: Optional[str] = None,
    display_plot: bool = True,
    figure_size: Optional[List[int]] = None,
    **kwargs,
) -> None:
    """Convenience function to plot causal graphs. This function uses different backends based on what's
    available on the system. The best result is achieved when using Graphviz as the backend. This requires both
    the Python pygraphviz package (``pip install pygraphviz``) and the shared system library (e.g. ``brew install
    graphviz`` or ``apt-get install graphviz``). When graphviz is not available, it will fall back to the
    networkx backend.

    :param causal_graph: The graph to be plotted
    :param causal_strengths: An optional dictionary with Edge -> float entries.
    :param colors: An optional dictionary with color specifications for edges or nodes.
    :param filename: An optional filename if the output should be plotted into a file.
    :param display_plot: Optionally specify if the plot should be displayed or not (default to True).
    :param figure_size: A tuple to define the width and height (as a tuple) of the pyplot. This is used to parameter to
                        modify pyplot's 'figure.figsize' parameter. If None is given, the current/default value is used.
    :param kwargs: Remaining parameters will be passed through to the backend verbatim.

    **Example usage**::

    >>> plot(nx.DiGraph([('X', 'Y')])) # plots X -> Y
    >>> plot(nx.DiGraph([('X', 'Y')]), causal_strengths={('X', 'Y'): 0.43}) # annotates arrow with 0.43
    >>> plot(nx.DiGraph([('X', 'Y')]), colors={('X', 'Y'): 'red', 'X': 'green'}) # colors X -> Y red and X green
    """
    try:
        from dowhy.gcm.util.pygraphviz import _plot_causal_graph_graphviz

        try:
            _plot_causal_graph_graphviz(
                causal_graph,
                causal_strengths=causal_strengths,
                colors=colors,
                filename=filename,
                display_plot=display_plot,
                figure_size=figure_size,
                **kwargs,
            )
        except Exception as error:
            _logger.info(
                "There was an error when trying to plot the graph via graphviz, falling back to networkx "
                "plotting. If graphviz is not installed, consider installing it for better looking plots. The"
                " error is:" + str(error)
            )
            _plot_causal_graph_networkx(
                causal_graph,
                causal_strengths=causal_strengths,
                colors=colors,
                filename=filename,
                display_plot=display_plot,
                figure_size=figure_size,
                **kwargs,
            )

    except ImportError:
        _logger.info(
            "Pygraphviz installation not found, falling back to networkx plotting. "
            "For better looking plots, consider installing pygraphviz. Note This requires both the Python "
            "pygraphviz package (``pip install pygraphviz``) and the shared system library (e.g. "
            "``brew install graphviz`` or ``apt-get install graphviz``)"
        )
        _plot_causal_graph_networkx(
            causal_graph,
            causal_strengths=causal_strengths,
            colors=colors,
            filename=filename,
            display_plot=display_plot,
            figure_size=figure_size,
            **kwargs,
        )


def plot_adjacency_matrix(
    adjacency_matrix: pd.DataFrame, is_directed: bool, filename: Optional[str] = None, display_plot: bool = True
) -> None:
    plot(
        nx.from_pandas_adjacency(adjacency_matrix, nx.DiGraph() if is_directed else nx.Graph()),
        display_plot=display_plot,
        filename=filename,
    )


def _plot_causal_graph_networkx(
    causal_graph: nx.Graph,
    pydot_layout_prog: Optional[str] = None,
    causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
    colors: Optional[Dict[Union[Any, Tuple[Any, Any]], str]] = None,
    filename: Optional[str] = None,
    display_plot: bool = True,
    label_wrap_length: int = 3,
    figure_size: Optional[List[int]] = None,
) -> None:
    if "graph" not in causal_graph.graph:
        causal_graph.graph["graph"] = {"rankdir": "TD"}

    if pydot_layout_prog is not None:
        layout = nx_pydot.pydot_layout(causal_graph, prog=pydot_layout_prog)
    else:
        layout = nx.spring_layout(causal_graph)

    if causal_strengths is None:
        causal_strengths = {}
    else:
        causal_strengths = deepcopy(causal_strengths)
    if colors is None:
        colors = {}
    else:
        colors = deepcopy(colors)

    max_strength = 0.0
    for (source, target, strength) in causal_graph.edges(data="CAUSAL_STRENGTH", default=1):
        if (source, target) not in causal_strengths:
            causal_strengths[(source, target)] = strength
        max_strength = max(max_strength, abs(causal_strengths[(source, target)]))
        if (source, target) not in colors:
            colors[(source, target)] = "black"

    for edge in causal_graph.edges:
        if edge[0] == edge[1]:
            raise ValueError(
                "Node %s has a self-cycle, i.e. a node pointing to itself. Plotting self-cycles is "
                "currently only supported for plots using Graphviz! Consider installing the corresponding"
                "requirements." % edge[0]
            )

    # Wrapping labels if they are too long
    labels = {}
    for node in causal_graph.nodes:
        if node not in colors:
            colors[node] = "lightblue"
        node_name_splits = str(node).split(" ")
        for i in range(1, len(node_name_splits)):
            if len(node_name_splits[i - 1]) > label_wrap_length:
                node_name_splits[i] = "\n" + node_name_splits[i]
            else:
                node_name_splits[i] = " " + node_name_splits[i]

        labels[node] = "".join(node_name_splits)

    if figure_size is not None:
        org_fig_size = pyplot.rcParams["figure.figsize"]
        pyplot.rcParams["figure.figsize"] = figure_size

    figure = pyplot.figure()

    nx.draw(
        causal_graph,
        pos=layout,
        node_color=[colors[node] for node in causal_graph.nodes()],
        edge_color=[colors[(s, t)] for (s, t) in causal_graph.edges()],
        linewidths=0.25,
        labels=labels,
        font_size=8,
        font_weight="bold",
        node_size=2000,
        width=[_calc_arrow_width(causal_strengths[(s, t)], max_strength) for (s, t) in causal_graph.edges()],
    )

    if display_plot:
        pyplot.show()

    if figure_size is not None:
        pyplot.rcParams["figure.figsize"] = org_fig_size

    if filename is not None:
        figure.savefig(filename)


def _calc_arrow_width(strength: float, max_strength: float):
    return 0.2 + 4.0 * float(abs(strength)) / float(max_strength)


def bar_plot(
    values: Dict[str, float],
    uncertainties: Optional[Dict[str, Tuple[float, float]]] = None,
    ylabel: str = "",
    filename: Optional[str] = None,
    display_plot: bool = True,
    figure_size: Optional[List[int]] = None,
    bar_width: float = 0.8,
    xticks: List[str] = None,
    xticks_rotation: int = 90,
    sort_names: bool = True,
) -> None:
    """Convenience function to make a bar plot of the given values with uncertainty bars, if provided. Useful for all
    kinds of attribution results (including confidence intervals).

    :param values: A dictionary where the keys are the labels and the values are the values to be plotted.
    :param uncertainties: A dictionary of attributes to be added to the error bars.
    :param ylabel: The label for the y-axis.
    :param filename: An optional filename if the output should be plotted into a file.
    :param display_plot: Optionally specify if the plot should be displayed or not (default to True).
    :param figure_size: The size of the figure to be plotted.
    :param bar_width: The width of the bars.
    :param xticks: Explicitly specify the labels for the bars on the x-axis.
    :param xticks_rotation: Specify the rotation of the labels on the x-axis.
    :param sort_names: If True, the names in the plot are sorted alphabetically. If False, the order as given in values
                       are used.
    """
    if sort_names:
        values = {k: values[k] for k in sorted(values)}

    if uncertainties is None:
        uncertainties = {node: [values[node], values[node]] for node in values}

    figure, ax = pyplot.subplots(figsize=figure_size)
    ci_plus = [uncertainties[node][1] - values[node] for node in values.keys()]
    ci_minus = [values[node] - uncertainties[node][0] for node in values.keys()]
    yerr = np.array([ci_minus, ci_plus])
    yerr[abs(yerr) < 10**-7] = 0
    pyplot.bar(values.keys(), values.values(), yerr=yerr, ecolor="#1E88E5", color="#ff0d57", width=bar_width)
    pyplot.ylabel(ylabel)
    pyplot.xticks(rotation=xticks_rotation)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if xticks:
        pyplot.xticks(list(uncertainties.keys()), xticks)

    if display_plot:
        pyplot.show()

    if filename is not None:
        figure.savefig(filename)
