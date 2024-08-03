import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import image

_logger = logging.getLogger(__name__)


def plot(
    causal_graph: nx.Graph,
    layout_prog: Optional[str] = None,
    causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
    colors: Optional[Dict[Union[Any, Tuple[Any, Any]], str]] = None,
    filename: Optional[str] = None,
    display_plot: bool = True,
    figure_size: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> None:
    """Convenience function to plot causal graphs. This function uses different backends based on what's
    available on the system. The best result is achieved when using Graphviz as the backend. This requires both
    the shared system library (e.g. ``brew install graphviz`` or ``apt-get install graphviz``) and the Python pygraphviz
    package (``pip install pygraphviz``). When graphviz is not available, it will fall back to the networkx backend.

    :param causal_graph: The graph to be plotted
    :param layout_prog: Defines the layout type. If None is given, the 'dot' layout is used for graphviz plots and a
                        customized layout for networkx plots.
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
        from dowhy.utils.graphviz_plotting import plot_causal_graph_graphviz

        try:
            plot_causal_graph_graphviz(
                causal_graph,
                layout_prog=layout_prog,
                causal_strengths=causal_strengths,
                colors=colors,
                filename=filename,
                display_plot=display_plot,
                figure_size=figure_size,
                **kwargs,
            )
        except Exception as error:
            from dowhy.utils.networkx_plotting import plot_causal_graph_networkx

            _logger.info(
                "There was an error when trying to plot the graph via graphviz, falling back to networkx "
                "plotting. If graphviz is not installed, consider installing it for better looking plots. The"
                " error is:" + str(error)
            )

            plot_causal_graph_networkx(
                causal_graph,
                layout_prog=layout_prog,
                causal_strengths=causal_strengths,
                colors=colors,
                filename=filename,
                display_plot=display_plot,
                figure_size=figure_size,
                **kwargs,
            )

    except (ImportError, ModuleNotFoundError):
        from dowhy.utils.networkx_plotting import plot_causal_graph_networkx

        _logger.info(
            "Pygraphviz installation not found, falling back to networkx plotting. "
            "For better looking plots, consider installing pygraphviz. Note This requires both the Python "
            "pygraphviz package (``pip install pygraphviz``) and the shared system library (e.g. "
            "``brew install graphviz`` or ``apt-get install graphviz``)"
        )

        plot_causal_graph_networkx(
            causal_graph,
            layout_prog=layout_prog,
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
    sort_names: bool = False,
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

        if xticks is not None:
            xticks = sorted(xticks)

    if uncertainties is None:
        uncertainties = {node: [values[node], values[node]] for node in values}
    else:
        for node in values:
            if node not in uncertainties:
                uncertainties[node] = [values[node], values[node]]

    figure, ax = plt.subplots(figsize=figure_size)
    ci_plus = np.array([uncertainties[node][1] - values[node] for node in values.keys()])
    ci_minus = np.array([values[node] - uncertainties[node][0] for node in values.keys()])

    is_negative_yerr = np.logical_or(ci_plus < 0, ci_minus < 0)
    ci_plus[is_negative_yerr] = 0
    ci_minus[is_negative_yerr] = 0

    yerr = np.array([ci_minus, ci_plus])
    plt.bar(values.keys(), values.values(), yerr=yerr, ecolor="#1E88E5", color="#ff0d57", width=bar_width)
    plt.ylabel(ylabel)
    plt.xticks(rotation=xticks_rotation)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if xticks:
        plt.xticks(list(uncertainties.keys()), xticks)

    if display_plot:
        plt.show()

    if filename is not None:
        figure.savefig(filename)


def _calc_arrow_width(strength: float, max_strength: float):
    if max_strength == 0:
        return 4.1
    elif max_strength < 0:
        raise ValueError("Got a negative strength! The strength needs to be positive.")

    return 0.1 + 4.0 * float(abs(strength)) / float(max_strength)


def _plot_as_pyplot_figure(pygraphviz_graph: Any, figure_size: Optional[Tuple[int, int]] = None) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pygraphviz_graph.draw(tmp_dir_name + os.sep + "Graph.png")
        img = image.imread(tmp_dir_name + os.sep + "Graph.png")

        if figure_size is not None:
            org_fig_size = plt.rcParams["figure.figsize"]
            plt.rcParams["figure.figsize"] = figure_size

        plt.imshow(img)
        plt.axis("off")
        plt.show()

        if figure_size is not None:
            plt.rcParams["figure.figsize"] = org_fig_size


def pretty_print_graph(graph: nx.DiGraph) -> None:
    """
    Pretty print the graph edges with time lags.

    :param graph: The networkx graph.
    :type graph: networkx.Graph
    :return: None
    :rtype: None
    """
    print("\nGraph edges with time lags:")
    for edge in graph.edges(data=True):
        print(f"{edge[0]} -> {edge[1]} with time-lagged dependency {edge[2]['time_lag']}")
