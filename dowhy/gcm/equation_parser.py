import ast
import logging
import re

import networkx as nx
import numpy as np
import scipy.stats

from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution, ScipyDistribution, StructuralCausalModel
from dowhy.gcm.causal_mechanisms import StochasticModel
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors

STOCHASTIC_MODEL_TYPES = {
    "empirical": EmpiricalDistribution,
    "bayesiangaussianmixture": EmpiricalDistribution,
    "parametric": ScipyDistribution,
}
NOISE_MODEL_PATTERN = r"^\s*([\w]+)\(([^)]*)\)\s*$"
banned_characters = [":", ";", "[", "__", "import", "lambda"]
allowed_callables = {}
np_functions = {func: getattr(np, func) for func in dir(np) if callable(getattr(np, func))}
scipy_functions = {
    func: getattr(scipy.stats, func) for func in dir(scipy.stats) if callable(getattr(scipy.stats, func))
}
builtin_functions = {"len": len, "__builtins__": {}}
allowed_callables.update(np_functions)
allowed_callables.update(scipy_functions)
allowed_callables.update(builtin_functions)

logger = logging.getLogger(__name__)


def create_causal_model_from_equations(node_equations: str):
    causal_nodes_info = {}
    causal_graph = nx.DiGraph()
    for equation in node_equations.split("\n"):
        equation = equation.strip()
        sanitize_input_expression(equation)
        if equation:
            node_name, expression = extract_equation_components(equation)
            check_node_redundancy(causal_nodes_info, node_name)
            if not (node_name in causal_nodes_info):
                causal_nodes_info[node_name] = {}
            root_node_match = re.match(NOISE_MODEL_PATTERN, expression)
            causal_graph.add_node(node_name)
            if root_node_match:
                causal_mechanism_name = root_node_match.group(1)
                args = root_node_match.group(2)
                parsed_args = parse_args(args) if args else {}
                causal_nodes_info[node_name]["causal_mechanism"] = identify_noise_model(
                    causal_mechanism_name, parsed_args
                )
            else:
                custom_func, noise_eq = expression.rsplit("+", 1)
                parent_nodes = extract_parent_nodes(custom_func)
                for parent_node in parent_nodes:
                    causal_graph.add_edge(parent_node, node_name)
                noise_model_name, parsed_args = extract_noise_model_components(noise_eq)
                noise_model = identify_noise_model(noise_model_name, parsed_args)
                causal_nodes_info[node_name]["causal_mechanism"] = AdditiveNoiseModel(
                    CustomModel(custom_func, parent_nodes), noise_model
                )
            causal_nodes_info[node_name]["fully_defined"] = True if parsed_args else False
    add_undefined_nodes_info(causal_nodes_info, causal_graph.nodes)
    causal_model = StructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        causal_model.set_causal_mechanism(node, causal_nodes_info[node]["causal_mechanism"])
        if causal_nodes_info[node]["fully_defined"]:
            causal_model.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(causal_model.graph, node)
    return causal_model


def parse_args(args: str):
    str_args_list = args.split(",")
    kwargs = {}
    for str_arg in str_args_list:
        if str_arg:
            arg_value_pairs = str_arg.split("=")
            kwargs[arg_value_pairs[0].strip()] = ast.literal_eval(arg_value_pairs[1].strip())
    return kwargs


def identify_noise_model(causal_mechanism_name: str, parsed_args: dict) -> StochasticModel:
    for model_type in STOCHASTIC_MODEL_TYPES:
        if model_type == causal_mechanism_name:
            return STOCHASTIC_MODEL_TYPES[model_type](**parsed_args)

    distribution = getattr(scipy.stats, causal_mechanism_name, None)
    if distribution:
        return STOCHASTIC_MODEL_TYPES["parametric"](scipy_distribution=distribution, **parsed_args)
    raise ValueError(f"Unable to recognise the noise model: {causal_mechanism_name}")


class CustomModel(PredictionModel):
    def __init__(self, custom_func: str, parent_nodes: list):
        self.custom_func = custom_func
        self.parent_nodes = parent_nodes

    def fit(self, X, Y):
        # Nothing to fit here, since we know the ground truth.
        pass

    def predict(self, X):
        local_dict = {self.parent_nodes[i]: X[:, i] for i in range(len(self.parent_nodes))}
        return shape_into_2d(eval(self.custom_func, allowed_callables, local_dict))
        # return shape_into_2d(ne.evaluate(self.custom_func, local_dict=local_dict, sanitize=True))

    def clone(self):
        return CustomModel(self.custom_func)


def extract_noise_model_components(noise_eq):
    noise_model_match = re.match(NOISE_MODEL_PATTERN, noise_eq)
    if noise_model_match:
        noise_model_name = noise_model_match.group(1)
        args = noise_model_match.group(2)
        parsed_args = parse_args(args)
        return noise_model_name, parsed_args
    else:
        raise Exception("Unable to recognise the format or function specified")


def extract_equation_components(equation):
    node_name, expression = equation.split("=", 1)
    node_name = node_name.strip()
    expression = expression.strip()
    return node_name, expression


def extract_parent_nodes(func_equation):
    parent_nodes = []
    # Find all node names in the expression string
    matched_node_names = re.findall(r"[A-Za-z_][a-zA-Z0-9_]*", func_equation)

    for matched_node in matched_node_names:
        if matched_node not in allowed_callables:
            parent_nodes.append(matched_node)
    parent_nodes.sort()
    return parent_nodes


def add_undefined_nodes_info(causal_nodes_info, present_nodes):
    for present_node in present_nodes:
        if present_node not in causal_nodes_info:
            logger.warning(f"{present_node} is undefined and will be considered as root node by default.")
            causal_nodes_info[present_node] = {}
            causal_nodes_info[present_node]["causal_mechanism"] = EmpiricalDistribution()
            causal_nodes_info[present_node]["fully_defined"] = False


def check_node_redundancy(causal_nodes_info, node_name):
    if node_name in causal_nodes_info:
        raise Exception(f"The node {node_name} is specified twice which is not allowed.")


def sanitize_input_expression(expression: str):
    for char in banned_characters:
        if char in expression:
            raise ValueError(f"'{char}' in the expression '{expression}' is not allowed because of security reasons")
    if re.search(r"[^0-9\+\-\*\/]+\.[^0-9\+\-\*\/]+", expression):
        raise ValueError(f"'.' can only be used incase of specifying decimals because of security reasons")
