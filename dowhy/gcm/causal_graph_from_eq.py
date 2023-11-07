import re
import ast
import networkx as nx
import numpy as np
import scipy.stats
from dowhy.gcm.causal_mechanisms import StochasticModel
from dowhy.gcm import EmpiricalDistribution, ScipyDistribution, StructuralCausalModel, AdditiveNoiseModel
from dowhy.gcm.ml.prediction_model import PredictionModel

_TYPES_OF_STOCHASTIC_MODELS = {"empirical": EmpiricalDistribution, "bayesiangaussianmixture": EmpiricalDistribution,
                               "parametric": ScipyDistribution}
stocastic_function_names = f'({"|".join(list(_TYPES_OF_STOCHASTIC_MODELS.keys()))})'
parent_node_pattern_for_eq = r'\b[A-Za-z_][A-Za-z_0-9 ]*\b'
# root_node_pattern = rf'^\s*(.+)\s*=\s*({stocastic_function_names})\(([^)]*)\)\s*$'
noise_model_pattern = rf'^\s*([\w]+)\(([^)]*)\)\s*$'


def create_causal_model_from_eq(node_equations: str):
    graph_node_pairs = []
    causal_nodes_info = {}
    for equation in node_equations.split('\n'):
        equation = equation.strip()
        if equation:
            node_name, expression = get_equation_components(equation)
            if not (node_name in causal_nodes_info):
                causal_nodes_info[node_name] = {}
            print("Variable Name:", node_name)
            root_node_match = re.match(noise_model_pattern, expression)
            if root_node_match:
                causal_mechanism_name = root_node_match.group(1)
                print(causal_mechanism_name)
                args = root_node_match.group(2)
                parsed_args = parse_args(args) if args else {}
                causal_nodes_info[node_name]['causal_mechanism'] = identify_noise_model(causal_mechanism_name,
                                                                                        parsed_args)
            else:
                custom_func, noise_eq = expression.rsplit('+', 1)
                graph_node_pairs += get_node_pairs(custom_func, node_name)
                noise_model_name, parsed_args = get_noise_model_components(noise_eq)
                noise_model = identify_noise_model(noise_model_name, parsed_args)
                causal_nodes_info[node_name]['causal_mechanism'] = AdditiveNoiseModel(MyCustomModel(custom_func),
                                                                                      noise_model)

    causal_graph = nx.DiGraph(graph_node_pairs)
    causal_model = StructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        causal_model.set_causal_mechanism(node, causal_nodes_info[node_name]['causal_mechanism'])
    return causal_model


def parse_args(args: str):
    print('args: ', args)
    str_args_list = args.split(',')
    kwargs = {}
    for str_arg in str_args_list:
        if str_arg:
            arg_value_pairs = str_arg.split('=')
            kwargs[arg_value_pairs[0]] = ast.literal_eval(arg_value_pairs[1])
    return kwargs


def identify_noise_model(causal_mechanism_name: str, parsed_args: dict) -> StochasticModel:
    for model_type in _TYPES_OF_STOCHASTIC_MODELS:
        if model_type == causal_mechanism_name:
            return _TYPES_OF_STOCHASTIC_MODELS[model_type](**parsed_args)
    return _TYPES_OF_STOCHASTIC_MODELS['parametric'](
        scipy_distribution=getattr(scipy.stats, causal_mechanism_name, None), **parsed_args)


class MyCustomModel(PredictionModel):
    def __init__(self, custom_func: str):
        self.custom_func = custom_func

    def fit(self, X, Y):
        # Nothing to fit here, since we know the ground truth.
        pass

    def predict(self, X):
        return ne.evaluate(custom_func, sanitize=True)

    def clone(self):
        return MyCustomModel(self.custom_func)


def get_noise_model_components(noise_eq):
    noise_model_match = re.match(noise_model_pattern, noise_eq)
    if noise_model_match:
        noise_model_name = noise_model_match.group(1)
        args = noise_model_match.group(2)
        parsed_args = parse_args(args)
        return noise_model_name, parsed_args
    else:
        raise InputError("The format of the equation entered should follow : F(X) + N")


def get_equation_components(equation):
    node_name, expression = equation.split("=", 1)
    node_name = node_name.strip()
    expression = expression.strip()
    return node_name, expression


def get_node_pairs(func_equation, child_node):
    node_pairs = []
    available_funcs = set(dir(__builtins__) + dir(np) + dir(scipy.stats))
    # Find all node names in the expression string
    parent_nodes = re.findall(parent_node_pattern_for_eq, func_equation)
    for parent_node in parent_nodes:
        if parent_node not in available_funcs:
            node_pairs.append((parent_node, child_node))
    return node_pairs