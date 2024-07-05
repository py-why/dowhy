import networkx as nx
import pandas as pd
from typing import List, Tuple

def find_lagged_parent_nodes(graph:nx.DiGraph, node:str) -> Tuple[List[str], List[int]]:
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
    if len(columns) != len(lag):
        raise ValueError("The size of 'columns' and 'lag' lists must be the same.")
    
    new_df = df.copy()
    for column, shift in zip(columns, lag):
        if shift > 0:
            new_df[column] = new_df[column].shift(shift, axis=0, fill_value=None)
    
    filled_df = new_df.fillna(0)
    return filled_df

def _filter_columns(df:pd.DataFrame, child_node, parent_nodes) -> pd.DataFrame:
    columns_to_keep = [str(child_node)] + list(parent_nodes.keys())
    filtered_df = df[columns_to_keep]
    return filtered_df
