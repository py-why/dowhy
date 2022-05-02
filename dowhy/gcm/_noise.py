import networkx as nx
import pandas as pd

from dowhy.gcm.cms import InvertibleStructuralCausalModel
from dowhy.gcm.graph import get_ordered_predecessors, is_root_node, validate_causal_dag
from dowhy.gcm.util.general import convert_numpy_array_to_pandas_column


def compute_noise_from_data(causal_model: InvertibleStructuralCausalModel, observed_data: pd.DataFrame) -> pd.DataFrame:
    validate_causal_dag(causal_model.graph)
    result = pd.DataFrame()

    for node in nx.topological_sort(causal_model.graph):
        if is_root_node(causal_model.graph, node):
            result[node] = observed_data[node]
        else:
            result[node] = convert_numpy_array_to_pandas_column(
                causal_model.causal_mechanism(node).estimate_noise(
                    observed_data[node].to_numpy(),
                    observed_data[get_ordered_predecessors(causal_model.graph, node)].to_numpy()))

    return result
