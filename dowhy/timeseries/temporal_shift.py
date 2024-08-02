from collections import deque
from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd


def add_lagged_edges(graph: nx.DiGraph, start_node: str) -> nx.DiGraph:
    """
    Perform a reverse BFS starting from the node and proceed to parents level-wise,
    adding edges from the ancestor to the current node with the accumulated time lag if it does not already exist.
    Additionally, create lagged nodes for each time lag encountered.

    :param graph: The directed graph object.
    :type graph: networkx.DiGraph
    :param start_node: The node from which to start the reverse BFS.
    :type start_node: string
    :return: A new graph with added edges based on accumulated time lags and lagged nodes.
    :rtype: networkx.DiGraph
    """
    new_graph = nx.DiGraph()
    queue = deque([start_node])
    lagged_node_mapping = {}  # Maps original nodes to their corresponding lagged nodes

    while queue:
        current_node = queue.popleft()

        for parent in graph.predecessors(current_node):
            edge_data = graph.get_edge_data(parent, current_node)
            if "time_lag" in edge_data:
                parent_time_lag = edge_data["time_lag"]

                # Ensure parent_time_lag is in tuple form
                if not isinstance(parent_time_lag, tuple):
                    parent_time_lag = (parent_time_lag,)

                for lag in parent_time_lag:
                    # Find or create the lagged node for the current node
                    if current_node in lagged_node_mapping:
                        lagged_nodes = lagged_node_mapping[current_node]
                    else:
                        lagged_nodes = set()
                        lagged_nodes.add(f"{current_node}_0")
                        new_graph.add_node(f"{current_node}_0")
                        lagged_node_mapping[current_node] = lagged_nodes

                    # For each lagged node, create new time-lagged parent nodes and add edges
                    new_lagged_nodes = set()
                    for lagged_node in lagged_nodes:
                        total_lag = -int(lagged_node.split("_")[-1]) + lag
                        new_lagged_parent_node = f"{parent}_{-total_lag}"
                        new_lagged_nodes.add(new_lagged_parent_node)

                        if not new_graph.has_node(new_lagged_parent_node):
                            new_graph.add_node(new_lagged_parent_node)

                        new_graph.add_edge(new_lagged_parent_node, lagged_node)

                        # Add the parent to the queue for further exploration
                        queue.append(parent)

                    # append the lagged nodes
                    if parent in lagged_node_mapping:
                        lagged_node_mapping[parent] = lagged_node_mapping[parent].union(new_lagged_nodes)
                    else:
                        lagged_node_mapping[parent] = new_lagged_nodes

        for original_node, lagged_nodes in lagged_node_mapping.items():
            sorted_lagged_nodes = sorted(lagged_nodes, key=lambda x: int(x.split("_")[-1]))
            for i in range(len(sorted_lagged_nodes) - 1):
                lesser_lagged_node = sorted_lagged_nodes[i]
                more_lagged_node = sorted_lagged_nodes[i + 1]
                new_graph.add_edge(lesser_lagged_node, more_lagged_node)

    return new_graph


def shift_columns_by_lag_using_unrolled_graph(df: pd.DataFrame, unrolled_graph: nx.DiGraph) -> pd.DataFrame:
    """
    Given a dataframe and an unrolled graph, this function shifts the columns in the dataframe by the corresponding time lags mentioned in the node names of the unrolled graph,
    creating a new unique column for each shifted version.

    :param df: The dataframe to shift.
    :type df: pandas.DataFrame
    :param unrolled_graph: The unrolled graph with nodes containing time lags in their names.
    :type unrolled_graph: networkx.DiGraph
    :return: The dataframe with the columns shifted by the corresponding time lags.
    :rtype: pandas.DataFrame
    """
    new_df = pd.DataFrame()
    for node in unrolled_graph.nodes:
        if "_" in node:
            base_node, lag_str = node.rsplit("_", 1)
            try:
                lag = -int(lag_str)
                if base_node in df.columns:
                    new_column_name = f"{base_node}_{-lag}"
                    new_df[new_column_name] = df[base_node].shift(lag, axis=0, fill_value=0)
            except ValueError:
                print(f"Warning: Cannot extract lag from node name {node}. Expected format 'baseNode_lag'")

    return new_df
