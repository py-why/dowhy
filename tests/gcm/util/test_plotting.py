import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.util import plot, plot_adjacency_matrix


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
