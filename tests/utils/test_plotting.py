import networkx as nx
import numpy as np
import pandas as pd
import pytest
from _pytest.python_api import approx

from dowhy.utils import plot, plot_adjacency_matrix
from dowhy.utils.networkx_plotting import plot_causal_graph_networkx
from dowhy.utils.plotting import _calc_arrow_width, bar_plot


def test_when_plot_does_not_raise_exception():
    plot(nx.DiGraph([("X", "Y"), ("Y", "Z")]))


def test_plot_adjacency_matrix():
    causal_graph = pd.DataFrame(
        np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), columns=["X", "Y", "Z"], index=["X", "Y", "Z"]
    )

    # Check if calling the method causes some import or runtime errors
    plot_adjacency_matrix(causal_graph, is_directed=True)
    # TODO: Plotting undirected graphs with networkx causes an error when an older networkx version is used with a newer
    #  newer matplotlib version:
    #  AttributeError: module 'matplotlib.cbook' has no attribute 'is_numlike'
    #  Networkx 2.4+ should fix this issue.
    # plot_adjacency_matrix(causal_graph, is_directed=False)


def test_given_causal_strengths_when_plot_graph_then_does_not_modify_input_object():
    causal_strength = {("X", "Y"): 10}

    plot(nx.DiGraph([("X", "Y"), ("Y", "Z")]), causal_strengths=causal_strength)

    assert causal_strength == {("X", "Y"): 10}


def test_given_colors_when_plot_graph_then_does_not_modify_input_object():
    colors = {("X", "Y"): "red", "X": "blue"}

    plot(nx.DiGraph([("X", "Y"), ("Y", "Z")]), colors=colors)

    assert colors == {("X", "Y"): "red", "X": "blue"}


def test_calc_arrow_width():
    assert _calc_arrow_width(0.4, max_strength=0.5) == approx(3.3, abs=0.01)
    assert _calc_arrow_width(0.2, max_strength=0.5) == approx(1.7, abs=0.01)
    assert _calc_arrow_width(-0.2, max_strength=0.5) == approx(1.7, abs=0.01)
    assert _calc_arrow_width(0.5, max_strength=0.5) == approx(4.1, abs=0.01)
    assert _calc_arrow_width(0.35, max_strength=0.5) == approx(2.9, abs=0.01)
    assert _calc_arrow_width(100, max_strength=101) == approx(4.06, abs=0.01)
    assert _calc_arrow_width(100, max_strength=0) == 4.1

    with pytest.raises(ValueError):
        _calc_arrow_width(100, max_strength=-1)


def test_given_misspecified_uncertainties_when_bar_plot_then_does_not_raise_error():
    bar_plot({"X": 1, "Y": 2, "Z": 3}, uncertainties={"X": (1.1, 0.9), "Y": (1.5, 2.1)})
    bar_plot({"X": 1, "Y": 2}, uncertainties={"X": (10**-2, 10**-1), "Y": (10**-2, 10**-1)})
    bar_plot({"X": 1}, uncertainties={"X": (0, 1)})
    bar_plot({"X": 1}, uncertainties={"X": (0, 0.99999999999)})
    bar_plot({"X": 1}, uncertainties={"X": (1 - 10**-15, 1 + 10**-15)})
