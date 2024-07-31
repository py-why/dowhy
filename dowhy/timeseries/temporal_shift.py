from typing import List, Optional, Tuple
from collections import deque

import networkx as nx
import pandas as pd


def find_ancestors(graph: nx.DiGraph, node: str) -> List[str]:
    """
    Given a graph and a node, this function returns the ancestor nodes of the node that are not parents.

    :param graph: The graph object.
    :type graph: networkx.Graph
    :param node: The node for which we want to find the ancestor nodes.
    :type node: string
    :return: A list of ancestor nodes of the node.
    :rtype: list
    """
    ancestors = []
    for n in nx.ancestors(graph, node):
        if n not in graph.predecessors(node):
            ancestors.append(n)
    return ancestors

# find parents and ancestors with accumulated time lags
def find_lagged_parents(graph: nx.DiGraph, node: str) -> Tuple[List[str], List[int]]:
    """
    Given a graph and a node, this function returns the parent nodes of the node and the time lags associated with the edges between the parent nodes and the node.

    :param graph: The graph object.
    :type graph: networkx.Graph
    :param node: The node for which we want to find the parent nodes.
    :type node: string
    :return: A tuple containing a list of parent nodes of the node and a list of time lags associated with the edges between the parent nodes and the node.
    :rtype: tuple (list, list)
    """
    parent_nodes = []
    time_lags = []
    for n in graph.predecessors(node):
        edge_data = graph.get_edge_data(n, node)
        if "time_lag" in edge_data:
            parent_nodes.append(n)
            time_lags.append(edge_data["time_lag"])
    return parent_nodes, time_lags

def reverse_bfs_with_accumulated_lags(graph: nx.DiGraph, start_node: str) -> nx.DiGraph:
    """
    Perform a reverse BFS starting from the node and proceed to parents level-wise,
    adding edges from the ancestor to the start_node with the accumulated time lag if it does not already exist.

    :param graph: The directed graph object.
    :type graph: networkx.DiGraph
    :param start_node: The node from which to start the reverse BFS.
    :type start_node: string
    :return: A new graph with added edges based on accumulated time lags.
    :rtype: networkx.DiGraph
    """
    new_graph = nx.DiGraph()
    queue = deque([(start_node, 0)])  # (current_node, accumulated_time_lag)
    visited = set()

    while queue:
        current_node, accumulated_time_lag = queue.popleft()
        
        if (current_node, accumulated_time_lag) in visited:
            continue

        visited.add((current_node, accumulated_time_lag))
        
        for parent in graph.predecessors(current_node):
            edge_data = graph.get_edge_data(parent, current_node)
            if "time_lag" in edge_data:
                parent_time_lag = edge_data["time_lag"]
                if isinstance(parent_time_lag, tuple):
                    for lag in parent_time_lag:
                        total_lag = accumulated_time_lag + lag
                        if not new_graph.has_edge(parent, start_node):
                            new_graph.add_edge(parent, start_node, time_lag=(total_lag,))
                        else:
                            existing_lags = new_graph[parent][start_node]["time_lag"]
                            new_graph[parent][start_node]["time_lag"] = existing_lags + (total_lag,)
                        queue.append((parent, total_lag))
                else:
                    total_lag = accumulated_time_lag + parent_time_lag
                    if not new_graph.has_edge(parent, start_node):
                        new_graph.add_edge(parent, start_node, time_lag=(total_lag,))
                    else:
                        existing_lags = new_graph[parent][start_node]["time_lag"]
                        new_graph[parent][start_node]["time_lag"] = existing_lags + (total_lag,)
                    queue.append((parent, total_lag))

    return new_graph

def shift_columns_by_lag(
    df: pd.DataFrame,
    columns: List[str],
    lag: List[int],
    ancestors: List[str],
    filter: bool,
    child_node: Optional[str] = None,
) -> pd.DataFrame:
    """
    Given a dataframe, a list of columns, and a list of time lags, this function shifts the columns in the dataframe by the corresponding time lags, creating a new unique column for each shifted version.
    Optionally, it can filter the dataframe to keep only the columns of the child node, the parent nodes, and their shifted versions.

    :param df: The dataframe to shift.
    :type df: pandas.DataFrame
    :param columns: A list of columns to shift.
    :type columns: list
    :param lag: A list of time lags to shift the columns by.
    :type lag: list
    :param ancestors: A list of ancestor nodes of the child node.
    :type ancestors: list
    :param filter: A boolean indicating whether to filter the dataframe to keep only relevant columns.
    :type filter: bool
    :param child_node: The child node to keep when filtering.
    :type child_node: int, optional
    :return: The dataframe with the columns shifted by the corresponding time lags and optionally filtered.
    :rtype: pandas.DataFrame
    """
    if len(columns) != len(lag):
        raise ValueError("The size of 'columns' and 'lag' lists must be the same.")

    new_df = df.copy()
    for column, max_lag in zip(columns, lag):
        for shift in range(1, max_lag + 1):
            new_column_name = f"{column}_lag{shift}"
            new_df[new_column_name] = new_df[column].shift(shift, axis=0, fill_value=0)

    if filter and child_node is not None:
        relevant_columns = (
            [child_node]
            + columns
            + [f"{col}_lag{shift}" for col in columns for shift in range(1, lag[columns.index(col)] + 1)]
        )
        relevant_columns = list(dict.fromkeys(relevant_columns))  # Ensure unique and maintain order
        new_df = new_df[relevant_columns]

    for ancestor in ancestors:
        new_df[ancestor] = df[ancestor]

    return new_df


def shift_columns_by_lag_using_unrolled_graph(
    df: pd.DataFrame,
    unrolled_graph: nx.DiGraph
) -> pd.DataFrame:
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
        if '_' in node:
            base_node, lag_str = node.rsplit('_', 1)
            try:
                lag = -int(lag_str)
                if base_node in df.columns:
                    new_column_name = f"{base_node}_{-lag}"
                    new_df[new_column_name] = df[base_node].shift(lag, axis=0, fill_value=0)
            except ValueError:
                print(f"Warning: Cannot extract lag from node name {node}. Expected format 'baseNode_lag'")
    
    return new_df