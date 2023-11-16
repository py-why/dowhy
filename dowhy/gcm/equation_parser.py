import re
import ast
import networkx as nx
import numexpr as ne
import numpy as np
import scipy.stats
from dowhy.gcm.causal_mechanisms import StochasticModel
from dowhy.gcm.util.general import shape_into_2d
from dowhy.gcm import EmpiricalDistribution, ScipyDistribution, StructuralCausalModel, AdditiveNoiseModel
from dowhy.gcm.ml.prediction_model import PredictionModel

STOCHASTIC_MODEL_TYPES = {"empirical": EmpiricalDistribution, "bayesiangaussianmixture": EmpiricalDistribution,
                               "parametric": ScipyDistribution}
NOISE_MODEL_PATTERN = rf'^\s*([\w]+)\(([^)]*)\)\s*$'


def create_causal_model_from_equations(node_equations: str):
    graph_node_pairs = []
    causal_nodes_info = {}
    for equation in node_equations.split('\n'):
        equation = equation.strip()
        if equation:
            node_name, expression = extract_equation_components(equation)
            if not (node_name in causal_nodes_info):
                causal_nodes_info[node_name] = {}
            print("Variable Name:", node_name)
            root_node_match = re.match(NOISE_MODEL_PATTERN, expression)
            if root_node_match:
                causal_mechanism_name = root_node_match.group(1)
                print(causal_mechanism_name)
                args = root_node_match.group(2)
                parsed_args = parse_args(args) if args else {}
                causal_nodes_info[node_name]['causal_mechanism'] = identify_noise_model(causal_mechanism_name,
                                                                                        parsed_args)
            else:
                custom_func, noise_eq = expression.rsplit('+', 1)
                parent_nodes = extract_parent_nodes(custom_func)
                graph_node_pairs += [(parent_node, node_name) for parent_node in parent_nodes]
                noise_model_name, parsed_args = extract_noise_model_components(noise_eq)
                noise_model = identify_noise_model(noise_model_name, parsed_args)
                causal_nodes_info[node_name]['causal_mechanism'] = AdditiveNoiseModel(MyCustomModel(custom_func, parent_nodes),
                                                                                      noise_model)

    causal_graph = nx.DiGraph(graph_node_pairs)
    causal_model = StructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        causal_model.set_causal_mechanism(node, causal_nodes_info[node]['causal_mechanism'])
    return causal_model


def parse_args(args: str):
    print('args: ', args)
    str_args_list = args.split(',')
    kwargs = {}
    for str_arg in str_args_list:
        if str_arg:
            arg_value_pairs = str_arg.split('=')
            kwargs[arg_value_pairs[0].strip()] = ast.literal_eval(arg_value_pairs[1].strip())
    return kwargs


def identify_noise_model(causal_mechanism_name: str, parsed_args: dict) -> StochasticModel:
    for model_type in STOCHASTIC_MODEL_TYPES:
        if model_type == causal_mechanism_name:
            return STOCHASTIC_MODEL_TYPES[model_type](**parsed_args)
    return STOCHASTIC_MODEL_TYPES['parametric'](
        scipy_distribution=getattr(scipy.stats, causal_mechanism_name, None), **parsed_args)


class MyCustomModel(PredictionModel):
    def __init__(self, custom_func: str, parent_nodes: list):
        self.custom_func = custom_func
        self.parent_nodes = parent_nodes

    def fit(self, X, Y):
        # Nothing to fit here, since we know the ground truth.
        pass

    def predict(self, X):
        local_dict = {self.parent_nodes[i]: X[:, i] for i in range(len(self.parent_nodes))}
        return shape_into_2d(ne.evaluate(self.custom_func, local_dict=local_dict,sanitize=True))

    def clone(self):
        return MyCustomModel(self.custom_func)


def extract_noise_model_components(noise_eq):
    noise_model_match = re.match(NOISE_MODEL_PATTERN, noise_eq)
    if noise_model_match:
        noise_model_name = noise_model_match.group(1)
        args = noise_model_match.group(2)
        parsed_args = parse_args(args)
        return noise_model_name, parsed_args
    else:
        raise ValueError("Unable to recognise the format or function specified")


def extract_equation_components(equation):
    node_name, expression = equation.split("=", 1)
    node_name = node_name.strip()
    expression = expression.strip()
    return node_name, expression


def extract_parent_nodes(func_equation):
    parent_nodes = []
    available_funcs = set(dir(__builtins__) + dir(np) + dir(scipy.stats))
    # Find all node names in the expression string
    matched_node_names = re.findall(r'\b[A-Za-z_][A-Za-z_0-9 ]*\b', func_equation)
    for matched_node in matched_node_names:
        if matched_node not in available_funcs:
            parent_nodes.append(matched_node)
    parent_nodes.sort()
    return parent_nodes