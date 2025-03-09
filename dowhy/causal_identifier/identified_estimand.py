import copy
from typing import List, Optional

import sympy as sp

from dowhy.utils.api import parse_state


class IdentifiedEstimand:
    """Class for storing a causal estimand, typically as a result of the identification step."""

    def __init__(
        self,
        identifier,  # This field will be deprecated in the future
        treatment_variable,
        outcome_variable,
        estimand_type=None,
        estimands=None,
        backdoor_variables=None,
        general_adjustment_variables=None,
        instrumental_variables=None,
        frontdoor_variables=None,
        mediator_variables=None,
        mediation_first_stage_confounders=None,
        mediation_second_stage_confounders=None,
        default_backdoor_id=None,
        default_adjustment_set_id=None,
        identifier_method=None,
        no_directed_path=False,
    ):
        self.identifier = identifier
        self.treatment_variable = parse_state(treatment_variable)
        self.outcome_variable = parse_state(outcome_variable)
        self.backdoor_variables = backdoor_variables
        self.general_adjustment_variables = general_adjustment_variables
        self.instrumental_variables = parse_state(instrumental_variables)
        self.frontdoor_variables = parse_state(frontdoor_variables)
        self.mediator_variables = parse_state(mediator_variables)
        self.mediation_first_stage_confounders = mediation_first_stage_confounders
        self.mediation_second_stage_confounders = mediation_second_stage_confounders
        self.estimand_type = estimand_type
        self.estimands = estimands
        self.default_backdoor_id = default_backdoor_id
        self.default_adjustment_set_id = default_adjustment_set_id
        self.identifier_method = identifier_method
        self.no_directed_path = no_directed_path

    def set_identifier_method(self, identifier_name: str):
        self.identifier_method = identifier_name

    def get_backdoor_variables(self, key: Optional[str] = None):
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

    def set_backdoor_variables(self, bdoor_variables_arr: List, key: Optional[str] = None):
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

    def get_general_adjustment_variables(self, key: Optional[str] = None):
        """Return a list containing general adjustment variables."""
        gav = self.general_adjustment_variables or {}
        return gav.get(self.default_adjustment_set_id if key is None else key, None)

    def set_general_adjustment_variables(self, variables_arr: List, key: Optional[str] = None):
        if key is None:
            key = self.identifier_method
        self.general_adjustment_variables[key] = variables_arr

    def get_adjustment_set(self, key: Optional[str] = None):
        if self.identifier_method == "general_adjustment":
            return self.get_general_adjustment_variables(key)
        return self.get_backdoor_variables(key)

    def set_adjustment_set(self, variables_arr: List, key: Optional[str] = None):
        if self.identifier_method == "general_adjustment":
            return self.set_general_adjustment_variables(variables_arr, key)
        return self.set_backdoor_variables(variables_arr, key)

    def __deepcopy__(self, memo):
        return IdentifiedEstimand(
            self.identifier,  # not deep copied
            copy.deepcopy(self.treatment_variable),
            copy.deepcopy(self.outcome_variable),
            estimand_type=copy.deepcopy(self.estimand_type),
            estimands=copy.deepcopy(self.estimands),
            backdoor_variables=copy.deepcopy(self.backdoor_variables),
            general_adjustment_variables=copy.deepcopy(self.general_adjustment_variables),
            instrumental_variables=copy.deepcopy(self.instrumental_variables),
            frontdoor_variables=copy.deepcopy(self.frontdoor_variables),
            mediator_variables=copy.deepcopy(self.mediator_variables),
            default_backdoor_id=copy.deepcopy(self.default_backdoor_id),
            default_adjustment_set_id=copy.deepcopy(self.default_adjustment_set_id),
            identifier_method=copy.deepcopy(self.identifier_method),
        )

    def __str__(self, only_target_estimand: bool = False, show_all_backdoor_sets: bool = False):
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
            # Just show the default generalized adjustment set
            if k.startswith("general") and k != "general_adjustment":
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
