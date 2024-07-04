import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def display_networkx_graph(graph):
    # Draw and display the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'time_lag')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()

def find_lagged_parent_nodes(graph, node):
    parent_nodes = {}
    for n in graph.predecessors(node):
        edge_data = graph.get_edge_data(n, node)
        if 'time_lag' in edge_data:
            parent_nodes[str(n)] = edge_data['time_lag']
    return parent_nodes

# once we have the parent dictionary then we can parse it and shift columns within the dataframe with the appropriate lag
def shift_columns(df:pd.DataFrame, parents:dict) -> pd.DataFrame:
    # rename parents to columns and allow lag parameter
    new_df = df.copy()
    for column, shift in parents.items():
        column=str(column)
        if shift > 0:
            new_df[column] = new_df[column].shift(shift, axis=0, fill_value=None)
    filled_df=new_df.fillna(0)
    return filled_df

def filter_columns(df:pd.DataFrame, child_node, parent_nodes) -> pd.DataFrame:
    columns_to_keep = [str(child_node)] + list(parent_nodes.keys())
    filtered_df = df[columns_to_keep]
    return filtered_df
