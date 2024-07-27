import networkx as nx
import pandas as pd
from typing import List, Tuple, Optional

def find_lagged_parents(graph:nx.DiGraph, node:str) -> Tuple[List[str], List[int]]:
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
        if 'time_lag' in edge_data:
            parent_nodes.append(n)
            time_lags.append(edge_data['time_lag'])
    return parent_nodes, time_lags

def shift_columns_by_lag(df: pd.DataFrame, columns: List[str], lag: List[int], filter: bool, child_node: Optional[str] = None) -> pd.DataFrame:
    """
    Given a dataframe, a list of columns, and a list of time lags, this function shifts the columns in the dataframe by the corresponding time lags, creating a new unique column for each shifted version.
    Optionally, it can filter the dataframe to keep only the columns of the child node, the parent nodes, and their shifted versions.

    :param df: The dataframe to shift.
    :type df: pandas.DataFrame
    :param columns: A list of columns to shift.
    :type columns: list
    :param lags: A list of time lags to shift the columns by.
    :type lags: list
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
        max_lag = int(max_lag)
        for shift in range(1, max_lag + 1):
            new_column_name = f"{column}_lag{shift}"
            new_df[new_column_name] = new_df[column].shift(shift, axis=0, fill_value=0)
    
    if filter and child_node is not None:
        relevant_columns = [child_node] + columns + [f"{col}_lag{shift}" for col in columns for shift in range(1, lag[columns.index(col)] + 1)]
        relevant_columns = list(dict.fromkeys(relevant_columns))  # Ensure unique and maintain order
        new_df = new_df[relevant_columns]
    
    return new_df
