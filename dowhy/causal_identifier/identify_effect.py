import copy
from enum import Enum
from typing import List, Protocol, Union

import sympy as sp

from dowhy.causal_graph import CausalGraph
from dowhy.utils.api import parse_state


class EstimandType(Enum):
    # Average total effect
    NONPARAMETRIC_ATE = "nonparametric-ate"
    # Natural direct effect
    NONPARAMETRIC_NDE = "nonparametric-nde"
    # Natural indirect effect
    NONPARAMETRIC_NIE = "nonparametric-nie"
    # Controlled direct effect
    NONPARAMETRIC_CDE = "nonparametric-cde"


class CausalIdentifier(Protocol):
    """
    Protocol to define a CausalIdentifier, all CausalIdentifiers must conform to at least this list of methods.
    """

    def identify_effect(
        self, graph: CausalGraph, treatment_name: Union[str, List[str]], outcome_name: Union[str, List[str]], **kwargs
    ):
        """Identify the causal effect to be estimated based on a CausalGraph
        :param graph: CausalGraph to be analyzed
        :param treatment_name: name of the treatment
        :param outcome_name: name of the outcome
        :param **kwargs: Additional parameters required by the identify_effect of a specific CausalIdentifier
        for example: conditional_node_names in AutoIdentifier or node_names in IDIdentifier
        :returns: a probability expression (estimand) for the causal effect if identified, else NULL
        """
        ...


class IdentifiedEstimand:

    """Class for storing a causal estimand, typically as a result of the identification step."""

    def __init__(
        self,
        identifier,
        treatment_variable,
        outcome_variable,
        estimand_type=None,
        estimands=None,
        backdoor_variables=None,
        instrumental_variables=None,
        frontdoor_variables=None,
        mediator_variables=None,
        mediation_first_stage_confounders=None,
        mediation_second_stage_confounders=None,
        default_backdoor_id=None,
        identifier_method=None,
        no_directed_path=False,
    ):
        self.identifier = identifier
        self.treatment_variable = parse_state(treatment_variable)
        self.outcome_variable = parse_state(outcome_variable)
        self.backdoor_variables = backdoor_variables
        self.instrumental_variables = parse_state(instrumental_variables)
        self.frontdoor_variables = parse_state(frontdoor_variables)
        self.mediator_variables = parse_state(mediator_variables)
        self.mediation_first_stage_confounders = mediation_first_stage_confounders
        self.mediation_second_stage_confounders = mediation_second_stage_confounders
        self.estimand_type = estimand_type
        self.estimands = estimands
        self.default_backdoor_id = default_backdoor_id
        self.identifier_method = identifier_method
        self.no_directed_path = no_directed_path

    def set_identifier_method(self, identifier_name):
        self.identifier_method = identifier_name

    def get_backdoor_variables(self, key=None):
        """Return a list containing the backdoor variables.

        If the calling estimator method is a backdoor method, return the
        backdoor variables corresponding to its target estimand.
        Otherwise, return the backdoor variables for the default backdoor estimand.
        """
        if key is None:
            if self.identifier_method and self.identifier_method.startswith("backdoor"):
                return self.backdoor_variables[self.identifier_method]
            elif self.backdoor_variables is not None and len(self.backdoor_variables) > 0:
                return self.backdoor_variables[self.default_backdoor_id]
            else:
                return []
        else:
            return self.backdoor_variables[key]

    def set_backdoor_variables(self, bdoor_variables_arr, key=None):
        if key is None:
            key = self.identifier_method
        self.backdoor_variables[key] = bdoor_variables_arr

    def get_frontdoor_variables(self):
        """Return a list containing the frontdoor variables (if present)"""
        return self.frontdoor_variables

    def get_mediator_variables(self):
        """Return a list containing the mediator variables (if present)"""
        return self.mediator_variables

    def get_instrumental_variables(self):
        """Return a list containing the instrumental variables (if present)"""
        return self.instrumental_variables

    def __deepcopy__(self, memo):
        return IdentifiedEstimand(
            self.identifier,  # not deep copied
            copy.deepcopy(self.treatment_variable),
            copy.deepcopy(self.outcome_variable),
            estimand_type=copy.deepcopy(self.estimand_type),
            estimands=copy.deepcopy(self.estimands),
            backdoor_variables=copy.deepcopy(self.backdoor_variables),
            instrumental_variables=copy.deepcopy(self.instrumental_variables),
            frontdoor_variables=copy.deepcopy(self.frontdoor_variables),
            mediator_variables=copy.deepcopy(self.mediator_variables),
            default_backdoor_id=copy.deepcopy(self.default_backdoor_id),
            identifier_method=copy.deepcopy(self.identifier_method),
        )

    def __str__(self, only_target_estimand=False, show_all_backdoor_sets=False):
        if self.no_directed_path:
            s = "No directed path from {0} to {1} in the causal graph.".format(
                self.treatment_variable, self.outcome_variable
            )
            s += "\nCausal effect is zero."
            return s
        s = "Estimand type: {0}\n".format(self.estimand_type)
        i = 1
        has_valid_backdoor = sum("backdoor" in key for key in self.estimands.keys())
        for k, v in self.estimands.items():
            if show_all_backdoor_sets:
                # Do not show backdoor key unless it is the only backdoor set.
                if k == "backdoor" and has_valid_backdoor > 1:
                    continue
            else:
                # Just show the default backdoor set
                if k.startswith("backdoor") and k != "backdoor":
                    continue
            if only_target_estimand and k != self.identifier_method:
                continue
            s += "\n### Estimand : {0}\n".format(i)
            s += "Estimand name: {0}".format(k)
            if k == self.default_backdoor_id:
                s += " (Default)"
            s += "\n"
            if v is None:
                s += "No such variable(s) found!\n"
            else:
                sp_expr_str = sp.pretty(v["estimand"], use_unicode=True)
                s += "Estimand expression:\n{0}\n".format(sp_expr_str)
                j = 1
                for ass_name, ass_str in v["assumptions"].items():
                    s += "Estimand assumption {0}, {1}: {2}\n".format(j, ass_name, ass_str)
                    j += 1
            i += 1
        return s


def identify_effect(
    graph: CausalGraph,
    treatment: Union[str, List[str]],
    outcome: Union[str, List[str]],
    method: CausalIdentifier,
    node_names=None,
    conditional_node_names=None,
):
    """Identify the causal effect to be estimated based on a CausalGraph

    :param graph: CausalGraph to be analyzed
    :param treatment: name of the treatment
    :param outcome: name of the outcome
    :param method: CausalIdentifier instance to use to identify effects
    :param node_names: OrderedSet comprising names of all nodes in the graph (Used for IDIdentifier only)
    :param conditional_node_names: variables that are used to determine treatment. If none are
    provided, it is assumed that the intervention is static (Used for AutoIdentifier only).
    :returns: a probability expression (estimand) for the causal effect if identified, else NULL
    """
    identified_estimand = method.identify_effect(
        graph, treatment, outcome, node_names=node_names, conditional_node_names=conditional_node_names
    )
    return identified_estimand
