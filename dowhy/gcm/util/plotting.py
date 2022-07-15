import logging
from typing import Optional, Dict, Tuple, Any

import networkx as nx
import pandas as pd
from matplotlib import pyplot
from networkx.drawing import nx_pydot

_logger = logging.getLogger(__name__)


def plot(causal_graph: nx.Graph,
         causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
         filename: Optional[str] = None,
         display_plot: bool = True,
         **kwargs) -> None:
    """Convenience function to plot causal graphs. This function uses different backends based on what's
    available on the system. The best result is achieved when using Graphviz as the backend. This requires both
    the Python pygraphviz package (``pip install pygraphviz``) and the shared system library (e.g. ``brew install
    graphviz`` or ``apt-get install graphviz``). When graphviz is not available, it will fall back to the
    networkx backend.

    :param causal_graph: The graph to be plotted
    :param causal_strengths: An optional dictionary with Edge -> float entries.
    :param filename: An optional filename if the output should be plotted into a file.
    :param display_plot: Optionally specify if the plot should be displayed or not (default to True).
    :param kwargs: Remaining parameters will be passed through to the backend verbatim.

    **Example usage**::

    >>> plot(nx.DiGraph([('X', 'Y')])) # plots X -> Y
    >>> plot(nx.DiGraph([('X', 'Y')]), causal_strengths={('X', 'Y'): 0.43}) # annotates arrow with 0.43
    """
    try:
        from dowhy.gcm.util.pygraphviz import _plot_causal_graph_graphviz
        try:
            _plot_causal_graph_graphviz(causal_graph,
                                        causal_strengths=causal_strengths,
                                        filename=filename,
                                        display_plot=display_plot,
                                        **kwargs)
        except Exception as error:
            _logger.info("There was an error when trying to plot the graph via graphviz, falling back to networkx "
                         "plotting. If graphviz is not installed, consider installing it for better looking plots. The"
                         " error is:" + str(error))
            _plot_causal_graph_networkx(causal_graph,
                                        causal_strengths=causal_strengths,
                                        filename=filename,
                                        display_plot=display_plot,
                                        **kwargs)

    except ImportError:
        _logger.info("Pygraphviz installation not found, falling back to networkx plotting. "
                     "For better looking plots, consider installing pygraphviz. Note This requires both the Python "
                     "pygraphviz package (``pip install pygraphviz``) and the shared system library (e.g. "
                     "``brew install graphviz`` or ``apt-get install graphviz``)")
        _plot_causal_graph_networkx(causal_graph,
                                    causal_strengths=causal_strengths,
                                    filename=filename,
                                    display_plot=display_plot,
                                    **kwargs)


def plot_adjacency_matrix(adjacency_matrix: pd.DataFrame,
                          is_directed: bool,
                          filename: Optional[str] = None,
                          display_plot: bool = True) -> None:
    plot(nx.from_pandas_adjacency(adjacency_matrix, nx.DiGraph() if is_directed else nx.Graph()),
         display_plot=display_plot,
         filename=filename)


def _plot_causal_graph_networkx(causal_graph: nx.Graph,
                                pydot_layout_prog: Optional[str] = None,
                                causal_strengths: Optional[Dict[Tuple[Any, Any], float]] = None,
                                filename: Optional[str] = None,
                                display_plot: bool = True,
                                label_wrap_length: int = 3) -> None:
    if 'graph' not in causal_graph.graph:
        causal_graph.graph['graph'] = {'rankdir': 'TD'}

    if pydot_layout_prog is not None:
        layout = nx_pydot.pydot_layout(causal_graph, prog=pydot_layout_prog)
    else:
        layout = nx.spring_layout(causal_graph)

    if causal_strengths is None:
        causal_strengths = {}

    max_strength = 0.0
    for (source, target, strength) in causal_graph.edges(data="CAUSAL_STRENGTH", default=1):
        if (source, target) not in causal_strengths:
            causal_strengths[(source, target)] = strength
        max_strength = max(max_strength, abs(causal_strengths[(source, target)]))

    for edge in causal_graph.edges:
        if edge[0] == edge[1]:
            raise ValueError("Node %s has a self-cycle, i.e. a node pointing to itself. Plotting self-cycles is "
                             "currently only supported for plots using Graphviz! Consider installing the corresponding"
                             "requirements." % edge[0])

    # Wrapping labels if they are too long
    labels = {}
    for node in causal_graph.nodes:
        node_name_splits = str(node).split(' ')
        for i in range(1, len(node_name_splits)):
            if len(node_name_splits[i - 1]) > label_wrap_length:
                node_name_splits[i] = '\n' + node_name_splits[i]
            else:
                node_name_splits[i] = ' ' + node_name_splits[i]

        labels[node] = ''.join(node_name_splits)

    figure = pyplot.figure()
    nx.draw(causal_graph,
            pos=layout,
            node_color='lightblue',
            linewidths=0.25,
            labels=labels,
            font_size=8,
            font_weight='bold',
            node_size=2000,
            width=[_calc_arrow_width(causal_strengths[(s, t)], max_strength)
                   for (s, t) in causal_graph.edges()])

    if filename is not None:
        figure.savefig(filename)

    if display_plot:
        pyplot.show()


def _calc_arrow_width(strength: float, max_strength: float):
    return 0.2 + 4.0 * float(abs(strength)) / float(max_strength)
