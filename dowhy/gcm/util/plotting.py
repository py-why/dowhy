import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

from dowhy.utils import plotting

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
    """Deprecated, please use dowhy.utils.plotting.plot() instead."""
    warnings.warn(
        "The plot method is deprecated. Use the plot function from dowhy.utils.plotting instead!", DeprecationWarning
    )
    plotting.plot(
        causal_graph=causal_graph,
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
    """Deprecated, please use dowhy.utils.plotting.plot_adjacency_matrix() instead."""
    warnings.warn(
        "The plot method is deprecated. Use the plot_adjacency_matrix function from dowhy.utils.plotting instead!",
        DeprecationWarning,
    )
    plotting.plot_adjacency_matrix(
        adjacency_matrix=adjacency_matrix, is_directed=is_directed, filename=filename, display_plot=display_plot
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
    sort_names: bool = True,
) -> None:
    """Deprecated, please use dowhy.utils.plotting.bar_plot() instead."""
    warnings.warn(
        "The plot method is deprecated. Use the bar_plot function from dowhy.utils.plotting instead!",
        DeprecationWarning,
    )
    plotting.bar_plot(
        values=values,
        uncertainties=uncertainties,
        ylabel=ylabel,
        filename=filename,
        display_plot=display_plot,
        figure_size=figure_size,
        bar_width=bar_width,
        xticks=xticks,
        xticks_rotation=xticks_rotation,
        sort_names=sort_names,
    )
