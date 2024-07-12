import networkx as nx
import pandas as pd
from typing import List, Tuple

def find_lagged_parent_nodes(graph:nx.DiGraph, node:str) -> Tuple[List[str], List[int]]:
    '''
    Given a graph and a node, this function returns the parent nodes of the node and the time lags associated with the edges between the parent nodes and the node.

    Args:
    - graph: the graph object
    - node: the node for which we want to find the parent nodes

    Returns:
    - parent_nodes: a list of parent nodes of the node
    - time_lags: a list of time lags associated with the edges between the parent nodes and the node
    '''
    parent_nodes = []
    time_lags = []
    for n in graph.predecessors(node):
        edge_data = graph.get_edge_data(n, node)
        if 'time_lag' in edge_data:
            parent_nodes.append(n)
            time_lags.append(edge_data['time_lag'])
    return parent_nodes, time_lags

# once we have the parent dictionary then we can parse it and shift columns within the dataframe with the appropriate lag
def shift_columns(df: pd.DataFrame, columns: List[str], lag: List[int]) -> pd.DataFrame:
    '''
    Given a dataframe, a list of columns and a list of time lags, this function shifts the columns in the dataframe by the corresponding time lags.
    
    Args:
    - df: the dataframe
    - columns: a list of columns to shift
    - lag: a list of time lags to shift the columns by

    Returns:
    - filled_df: the dataframe with the columns shifted by the corresponding
    '''
    if len(columns) != len(lag):
        raise ValueError("The size of 'columns' and 'lag' lists must be the same.")
    
    new_df = df.copy()
    for column, shift in zip(columns, lag):
        if shift > 0:
            new_df[str(column)] = new_df[str(column)].shift(shift, axis=0, fill_value=None)
    
    filled_df = new_df.fillna(0)
    return filled_df

def _filter_columns(df:pd.DataFrame, child_node:int, parent_nodes:List[int]) -> pd.DataFrame:
    '''
    Given a dataframe, a child node and a dictionary of parent nodes, this function filters the dataframe to keep only the columns of the child node and the parent nodes.

    Args:
    - df: the dataframe
    - child_node: the child node
    - parent_nodes: a list of parent nodes

    Returns:
    - filtered_df: the dataframe with only the columns of the child node and the parent nodes
    '''
    columns_to_keep = [str(node) for node in parent_nodes]
    columns_to_keep += [str(child_node)] 
    filtered_df = df[columns_to_keep]
    return filtered_df
