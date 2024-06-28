import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def create_graph_from_user():
    # Initialize an empty directed graph
    graph = nx.DiGraph()
    
    # Instructions for the user
    print("Enter the graph as a list of edges with time lags. Enter 'done' when you are finished.")
    print("Each edge should be entered in the format 'node1 node2 time_lag'. For example: 'A B 4'")
    
    # Loop to receive user input
    while True:
        edge = input("Enter an edge: ")
        if edge.lower() == "done":
            break
        edge = edge.split()
        if len(edge) != 3:
            print("Invalid edge. Please enter an edge in the format 'node1 node2 time_lag'.")
            continue
        node1, node2, time_lag = edge
        try:
            time_lag = float(time_lag)
        except ValueError:
            print("Invalid weight. Please enter a numerical value for the time_lag.")
            continue
        graph.add_edge(node1, node2, time_lag=time_lag)
    
    return graph

def create_graph_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Initialize an empty directed graph
    graph = nx.DiGraph()
    
    # Add edges with weights to the graph
    for index, row in df.iterrows():
        graph.add_edge(row['node1'], row['node2'], time_lag=row['time_lag'])
    
    return graph

def pretty_print_graph(graph):
    # Display the entered graph
    print("\nGraph edges with time lags:")
    for edge in graph.edges(data=True):
        print(f"{edge[0]} -> {edge[1]} with time-lagged dependency {edge[2]['time_lag']}")

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
def shift_columns(df, parents):
    new_df = df.copy()
    for column, shift in parents.items():
        column=str(column)
        if shift > 0:
            new_df[column] = new_df[column].shift(shift, axis=0, fill_value=None)
        # elif shift < 0:
        #     new_df[column] = new_df[column].shift(shift, axis=0, fill_value=None)
        #     new_df.drop(new_df.index[0:abs(shift)], axis=0, inplace=True)
    return new_df

def filter_columns(df, child_node, parent_nodes):
    columns_to_keep = [str(child_node)] + list(parent_nodes.keys())
    filtered_df = df[columns_to_keep]
    return filtered_df
