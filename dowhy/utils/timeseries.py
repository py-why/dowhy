import networkx as nx
import numpy as np
import pandas as pd


def create_graph_from_user() -> nx.DiGraph:
    """
    Creates a directed graph based on user input from the console.

    The time_lag parameter of the networkx graph represents the exact causal lag of an edge between any 2 nodes in the graph.
    Each edge can contain multiple time lags, therefore each combination of (node1,node2,time_lag) must be input individually by the user.

    The user is prompted to enter edges one by one in the format 'node1 node2 time_lag',
    where 'node1' and 'node2' are the nodes connected by the edge, and 'time_lag' is a numerical
    value representing the weight of the edge. The user should enter 'done' to finish inputting edges.

    :return: A directed graph created from the user's input.
    :rtype: nx.DiGraph

    Example user input:
        Enter an edge: A B 4
        Enter an edge: B C 2
        Enter an edge: done
    """
    graph = nx.DiGraph()

    print("Enter the graph as a list of edges with time lags. Enter 'done' when you are finished.")
    print("Each edge should be entered in the format 'node1 node2 time_lag'. For example: 'A B 4'")

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
            time_lag = int(time_lag)
        except ValueError:
            print("Invalid weight. Please enter a numerical value for the time_lag.")
            continue

        # Check if the edge already exists
        if graph.has_edge(node1, node2):
            # If the edge exists, append the time_lag to the existing tuple
            current_time_lag = graph[node1][node2]["time_lag"]
            if isinstance(current_time_lag, tuple):
                graph[node1][node2]["time_lag"] = current_time_lag + (time_lag,)
            else:
                graph[node1][node2]["time_lag"] = (current_time_lag, time_lag)
        else:
            # If the edge does not exist, create a new edge with a tuple containing the time_lag
            graph.add_edge(node1, node2, time_lag=(time_lag,))

    return graph


def create_graph_from_csv(file_path: str) -> nx.DiGraph:
    """
    Creates a directed graph from a CSV file.

    The time_lag parameter of the networkx graph represents the exact causal lag of an edge between any 2 nodes in the graph.
    Each edge can contain multiple time lags, therefore each combination of (node1,node2,time_lag) must be input individually in the CSV file.

    The CSV file should have at least three columns: 'node1', 'node2', and 'time_lag'.
    Each row represents an edge from 'node1' to 'node2' with a 'time_lag' attribute.

    :param file_path: The path to the CSV file.
    :type file_path: str
    :return: A directed graph created from the CSV file.
    :rtype: nx.DiGraph

    Example:
        Example CSV content:

        .. code-block:: csv

            node1,node2,time_lag
            A,B,5
            B,C,2
            A,C,7
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize an empty directed graph
    graph = nx.DiGraph()

    # Add edges with time lag to the graph
    for index, row in df.iterrows():
        # Add validation for the time lag column to be a number
        try:
            time_lag = int(row["time_lag"])
        except ValueError:
            print(
                "Invalid weight. Please enter a numerical value for the time_lag for the edge between {} and {}.".format(
                    row["node1"], row["node2"]
                )
            )
            return None

        # Check if the edge already exists
        if graph.has_edge(row["node1"], row["node2"]):
            # If the edge exists, append the time_lag to the existing tuple
            current_time_lag = graph[row["node1"]][row["node2"]]["time_lag"]
            if isinstance(current_time_lag, tuple):
                graph[row["node1"]][row["node2"]]["time_lag"] = current_time_lag + (time_lag,)
            else:
                graph[row["node1"]][row["node2"]]["time_lag"] = (current_time_lag, time_lag)
        else:
            # If the edge does not exist, create a new edge with a tuple containing the time_lag
            graph.add_edge(row["node1"], row["node2"], time_lag=(time_lag,))

    return graph


def create_graph_from_dot_format(file_path: str) -> nx.DiGraph:
    """
    Creates a directed graph from a DOT file and ensures it is a DiGraph.

    The time_lag parameter of the networkx graph represents the exact causal lag of an edge between any 2 nodes in the graph.
    Each edge can contain multiple valid time lags.

    The DOT file should contain a graph in DOT format.

    :param file_path: The path to the DOT file.
    :type file_path: str
    :return: A directed graph (DiGraph) created from the DOT file.
    :rtype: nx.DiGraph
    """
    # Read the DOT file into a MultiDiGraph
    multi_graph = nx.drawing.nx_agraph.read_dot(file_path)

    # Initialize a new DiGraph
    graph = nx.DiGraph()

    # Iterate over edges of the MultiDiGraph
    for u, v, data in multi_graph.edges(data=True):
        if "label" in data:
            try:
                # Convert the label to a tuple of time lags
                time_lag_tuple = tuple(map(int, data["label"].strip("()").split(",")))

                if graph.has_edge(u, v):
                    existing_data = graph.get_edge_data(u, v)
                    if "time_lag" in existing_data:
                        # Merge the existing time lags with the new ones
                        existing_time_lags = existing_data["time_lag"]
                        new_time_lags = existing_time_lags + time_lag_tuple
                        # Remove duplicates by converting to a set and back to a tuple
                        graph[u][v]["time_lag"] = tuple(set(new_time_lags))
                    else:
                        graph[u][v]["time_lag"] = time_lag_tuple
                else:
                    graph.add_edge(u, v, time_lag=time_lag_tuple)

            except ValueError:
                print(f"Invalid weight for the edge between {u} and {v}.")
                return None

    return graph


def create_graph_from_networkx_array(array: np.ndarray, var_names: list) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from a numpy array with time lag information.

    The time_lag parameter of the networkx graph represents the exact causal lag of an edge between any 2 nodes in the graph.
    Each edge can contain multiple valid time lags.

    The resulting graph will be a directed graph with edge attributes indicating
    the type of link based on the array values.

    :param array: A numpy array of shape (n, n, tau) representing the causal links.
    :type array: np.ndarray
    :param var_names: A list of variable names.
    :type var_names: list
    :return: A directed graph with edge attributes based on the array values.
    :rtype: nx.DiGraph
    """
    n = array.shape[0]  # Number of variables
    assert n == array.shape[1], "The array must be square."
    tau = array.shape[2]  # Number of time lags

    # Initialize a directed graph
    graph = nx.DiGraph()

    # Add nodes with names
    graph.add_nodes_from(var_names)

    # Iterate over all pairs of nodes
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # Skip self-loops

            for t in range(tau):
                # Check for directed links
                if array[i, j, t] == "-->":
                    if graph.has_edge(var_names[i], var_names[j]):
                        # Append the time lag to the existing tuple
                        current_time_lag = graph[var_names[i]][var_names[j]].get("time_lag", ())
                        graph[var_names[i]][var_names[j]]["time_lag"] = current_time_lag + (t,)
                    else:
                        # Create a new edge with a tuple containing the time lag
                        graph.add_edge(var_names[i], var_names[j], time_lag=(t,))

                elif array[i, j, t] == "<--":
                    if graph.has_edge(var_names[j], var_names[i]):
                        # Append the time lag to the existing tuple
                        current_time_lag = graph[var_names[j]][var_names[i]].get("time_lag", ())
                        graph[var_names[j]][var_names[i]]["time_lag"] = current_time_lag + (t,)
                    else:
                        # Create a new edge with a tuple containing the time lag
                        graph.add_edge(var_names[j], var_names[i], time_lag=(t,))

                elif array[i, j, t] == "o-o":
                    raise ValueError(
                        "Unsupported link type 'o-o' found between {} and {} at lag {}.".format(
                            var_names[i], var_names[j], t
                        )
                    )

                elif array[i, j, t] == "x-x":
                    raise ValueError(
                        "Unsupported link type 'x-x' found between {} and {} at lag {}.".format(
                            var_names[i], var_names[j], t
                        )
                    )

    return graph
