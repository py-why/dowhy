import networkx as nx
import pandas as pd
from typing import List, Tuple

def find_lagged_parent_nodes(graph:nx.DiGraph, node:str) -> Tuple[List[str], List[int]]:
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

def shift_columns(df: pd.DataFrame, columns: List[str], lag: List[int]) -> pd.DataFrame:
    """
    Given a dataframe, a list of columns, and a list of time lags, this function shifts the columns in the dataframe by the corresponding time lags, creating a new unique column for each shifted version.

    :param df: The dataframe to shift.
    :type df: pandas.DataFrame
    :param columns: A list of columns to shift.
    :type columns: list
    :param lags: A list of time lags to shift the columns by.
    :type lags: list
    :return: The dataframe with the columns shifted by the corresponding time lags.
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
    
    return new_df

def _filter_columns(df: pd.DataFrame, child_node: int, parent_nodes: List[int]) -> pd.DataFrame:
    """
    Given a dataframe, a target node, and a list of action/parent nodes, this function filters the dataframe to keep only the columns of the target node, the parent nodes, and their shifted versions.

    :param df: The dataframe to filter.
    :type df: pandas.DataFrame
    :param child_node: The child node.
    :type child_node: int
    :param parent_nodes: A list of parent nodes.
    :type parent_nodes: list
    :return: The dataframe with only the columns of the child node, parent nodes, and their shifted versions.
    :rtype: pandas.DataFrame
    """
    columns_to_keep = [str(child_node)]
    for node in parent_nodes:
        columns_to_keep.append(str(node))
        # Include all shifted versions of the parent node
        shifted_columns = [col for col in df.columns if col.startswith(f"{node}_lag")]
        columns_to_keep.extend(shifted_columns)
    
    # Filter the dataframe to keep only the relevant columns
    filtered_df = df[columns_to_keep]
    return filtered_df