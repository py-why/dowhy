import ast
import logging
import re
from typing import Tuple

import networkx as nx
import numpy as np
import scipy.stats

from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution, ScipyDistribution, StructuralCausalModel
from dowhy.gcm.causal_mechanisms import StochasticModel
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors

_STOCHASTIC_MODEL_TYPES = {
    "empirical": EmpiricalDistribution,
    "bayesiangaussianmixture": EmpiricalDistribution,
    "parametric": ScipyDistribution,
}
_NOISE_MODEL_PATTERN = r"^\s*([\w]+)\(([^)]*)\)\s*$"
_NODE_NAME_PATTERN = r"[a-zA-Z_]\w*"
_UNKNOWN_MODEL_PATTERN = rf"\s*\b{_NODE_NAME_PATTERN}(?:\s*,\s*{_NODE_NAME_PATTERN})*\b"
_allowed_callables = {}
_np_functions = {func: getattr(np, func) for func in dir(np) if callable(getattr(np, func))}
_scipy_functions = {
    func: getattr(scipy.stats, func) for func in dir(scipy.stats) if callable(getattr(scipy.stats, func))
}
_builtin_functions = {"len": len, "__builtins__": {}}
_allowed_callables.update(_np_functions)
_allowed_callables.update(_scipy_functions)
_allowed_callables.update(_builtin_functions)

logger = logging.getLogger(__name__)


def create_causal_model_from_equations(node_equations: str) -> StructuralCausalModel:
    """
    Create a causal model from a set of equations defining causal relationships between nodes.
    The equation format supports the following cases in which expression can be defined:
    1. Specifying root node equation:
        >>> "<node_name> = <noise_model_name>(<optional_arguments>)"
    The noise model name can be one of the following:
        - empirical()
        - bayesiangaussianmixture()
        - parametric()
        - <scipy.stats.*>
    Empirical and bayessian models are already defined and one can find the description
    of those in the dowhy library.
    Use parametric when you want to find the best continuous distribution for the data.
    You can specify any noise function defined in scipy\.stats library.
    2. Specifying non-root node equation:
        >>> "<node_name> = <function-expression> + <noise_model_name>(<optional_arguments>)"
    The function-expression can be any expression containing airthmetic operations of the nodes
    and calling functions defined under numpy. The format/definition of noise for the non-root node
    remains same as in point one.
    3. Specifying unknown causal model equation:
        >>> "Node -> <node_name1>, <node_name2>, ..."
    In case we don't know the causal relationship model between nodes then we can
    use the above format to just define the edges between the nodes.
    Example:
        >>> scm = \"""
        X = empirical()
        Z = norm(loc=0, scale=1)
        Y = 12 * X + log(Z) + norm(loc=0, scale=1)
        \"""
    :param node_equations: A string containing equations defining the relationships between nodes.
                            Each equation should be separated by a newline.
    :return: StructuralCausalModel: A StructuralCausalModel object representing the created causal model.
    """
    banned_characters = [":", ";", "[", "__", "import", "lambda"]
    causal_nodes_info = {}
    causal_graph = nx.DiGraph()
    for equation in node_equations.split("\n"):
        equation = equation.strip()
        _sanitize_input_expression(equation, banned_characters)
        if equation:
            parsed_args = {}
            node_name, expression = _extract_equation_components(equation)
            _check_node_redundancy(causal_nodes_info, node_name)
            causal_nodes_info[node_name] = {}
            root_node_match = re.match(_NOISE_MODEL_PATTERN, expression)
            unknown_model_match = _check_if_model_is_unknown(equation, expression)
            causal_graph.add_node(node_name)
            if root_node_match:
                causal_mechanism_name = root_node_match.group(1)
                args = root_node_match.group(2)
                parsed_args = _parse_args(args)
                causal_nodes_info[node_name]["causal_mechanism"] = _identify_noise_model(
                    causal_mechanism_name, parsed_args
                )
            elif unknown_model_match:
                parent_node_candidates = expression.split(",")
                parent_nodes = _get_sorted_parent_nodes(parent_node_candidates)
                _add_parent_nodes_to_graph(causal_graph, parent_nodes, node_name)
                causal_nodes_info[node_name]["unknown"] = True
            else:
                custom_func, noise_eq = expression.rsplit("+", 1)
                # Find all node names in the expression string.
                parent_node_candidates = re.findall(_NODE_NAME_PATTERN, custom_func)
                parent_nodes = _get_sorted_parent_nodes(parent_node_candidates)
                _add_parent_nodes_to_graph(causal_graph, parent_nodes, node_name)
                noise_model_name, parsed_args = _extract_noise_model_components(noise_eq)
                noise_model = _identify_noise_model(noise_model_name, parsed_args)
                causal_nodes_info[node_name]["causal_mechanism"] = AdditiveNoiseModel(
                    CustomEquationModel(custom_func, parent_nodes), noise_model
                )
            causal_nodes_info[node_name]["fully_defined"] = True if parsed_args else False
    _add_undefined_nodes_info(causal_nodes_info, list(causal_graph.nodes))
    causal_model = StructuralCausalModel(causal_graph)
    for node in causal_graph.nodes:
        if not ("unknown" in causal_nodes_info[node]):
            causal_model.set_causal_mechanism(node, causal_nodes_info[node]["causal_mechanism"])
        if causal_nodes_info[node]["fully_defined"]:
            causal_model.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(causal_model.graph, node)
    return causal_model


def _parse_args(args: str) -> dict:
    str_args_list = args.split(",")
    kwargs = {}
    for str_arg in str_args_list:
        if str_arg:
            arg_value_pairs = str_arg.split("=")
            kwargs[arg_value_pairs[0].strip()] = ast.literal_eval(arg_value_pairs[1].strip())
    return kwargs


def _add_parent_nodes_to_graph(causal_graph: nx.DiGraph, parent_nodes: list, node_name: str) -> None:
    for parent_node in parent_nodes:
        causal_graph.add_edge(parent_node, node_name)


def _identify_noise_model(causal_mechanism_name: str, parsed_args: dict) -> StochasticModel:
    for model_type in _STOCHASTIC_MODEL_TYPES:
        if model_type == causal_mechanism_name:
            return _STOCHASTIC_MODEL_TYPES[model_type](**parsed_args)

    distribution = getattr(scipy.stats, causal_mechanism_name, None)
    if distribution:
        return _STOCHASTIC_MODEL_TYPES["parametric"](scipy_distribution=distribution, **parsed_args)
    raise ValueError(f"Unable to recognise the noise model: {causal_mechanism_name}")


def _extract_noise_model_components(noise_eq: str) -> Tuple[str, dict]:
    noise_model_match = re.match(_NOISE_MODEL_PATTERN, noise_eq)
    if noise_model_match:
        noise_model_name = noise_model_match.group(1)
        args = noise_model_match.group(2)
        parsed_args = _parse_args(args)
        return noise_model_name, parsed_args
    else:
        raise Exception("Unable to recognise the format or function specified")


def _extract_equation_components(equation: str) -> Tuple[str, str]:
    if "->" in equation:
        node_name, expression = equation.split("->", 1)
    else:
        node_name, expression = equation.split("=", 1)
    node_name = node_name.strip()
    expression = expression.strip()
    return node_name, expression


def _get_sorted_parent_nodes(parent_node_candidates: list) -> list:
    parent_nodes = []
    for candidate_node_name in parent_node_candidates:
        candidate_node_name = candidate_node_name.strip()
        if candidate_node_name not in _allowed_callables:
            parent_nodes.append(candidate_node_name)
    parent_nodes.sort()
    return parent_nodes


def _add_undefined_nodes_info(causal_nodes_info: dict, present_nodes: list) -> None:
    for present_node in present_nodes:
        if present_node not in causal_nodes_info:
            logger.warning(f"{present_node} is undefined and will be considered as root node by default.")
            causal_nodes_info[present_node] = {}
            causal_nodes_info[present_node]["causal_mechanism"] = EmpiricalDistribution()
            causal_nodes_info[present_node]["fully_defined"] = False


def _check_node_redundancy(causal_nodes_info: dict, node_name: str) -> None:
    if node_name in causal_nodes_info:
        raise Exception(f"The node {node_name} is specified twice which is not allowed.")


def _sanitize_input_expression(expression: str, banned_characters: list) -> None:
    for char in banned_characters:
        if char in expression:
            raise ValueError(f"'{char}' in the expression '{expression}' is not allowed because of security reasons")
    if re.search(r"[^0-9\+\-\*\/]+\.[^0-9\+\-\*\/]+", expression):
        raise ValueError(f"'.' can only be used incase of specifying decimals because of security reasons")


def _check_if_model_is_unknown(equation: str, expression: str) -> bool:
    if "->" in equation:
        if re.match(_UNKNOWN_MODEL_PATTERN, expression):
            return True
    return False


class CustomEquationModel(PredictionModel):
    """
    Represents custom prediction model implementation. This model does not require to be fitted as the model has to be fully defined.
    """

    def __init__(self, custom_func: str, parent_nodes: list):
        self.custom_func = custom_func
        self.parent_nodes = parent_nodes

    def fit(self, X, Y) -> None:
        # Nothing to fit here, since we know the ground truth.
        pass

    def predict(self, X) -> np.ndarray:
        local_dict = {self.parent_nodes[i]: X[:, i] for i in range(len(self.parent_nodes))}
        return shape_into_2d(eval(self.custom_func, _allowed_callables, local_dict))

    def clone(self):
        return CustomEquationModel(self.custom_func, self.parent_nodes)
