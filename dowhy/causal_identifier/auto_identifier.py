import itertools
import logging
import sys
from enum import Enum
from typing import Dict, List, Optional, Union

import networkx as nx
import sympy as sp
import sympy.stats as spstats

from dowhy.causal_identifier.adjustment_set import AdjustmentSet
from dowhy.causal_identifier.efficient_backdoor import EfficientBackdoor
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.graph import (
    check_dseparation,
    check_valid_backdoor_set,
    check_valid_frontdoor_set,
    check_valid_mediation_set,
    do_surgery,
    get_all_directed_paths,
    get_backdoor_paths,
    get_descendants,
    get_instruments,
    get_proper_backdoor_graph,
    get_proper_causal_path_nodes,
    has_directed_path,
)
from dowhy.utils.api import parse_state

logger = logging.getLogger(__name__)


class EstimandType(Enum):
    # Average total effect
    NONPARAMETRIC_ATE = "nonparametric-ate"
    # Natural direct effect
    NONPARAMETRIC_NDE = "nonparametric-nde"
    # Natural indirect effect
    NONPARAMETRIC_NIE = "nonparametric-nie"
    # Controlled direct effect
    NONPARAMETRIC_CDE = "nonparametric-cde"


class BackdoorAdjustment(Enum):
    # Backdoor method names
    BACKDOOR_DEFAULT = "default"
    BACKDOOR_EXHAUSTIVE = "exhaustive-search"
    BACKDOOR_MIN = "minimal-adjustment"
    BACKDOOR_MAX = "maximal-adjustment"
    BACKDOOR_EFFICIENT = "efficient-adjustment"
    BACKDOOR_MIN_EFFICIENT = "efficient-minimal-adjustment"
    BACKDOOR_MINCOST_EFFICIENT = "efficient-mincost-adjustment"


class GeneralizedAdjustment(Enum):
    # Covariate adjustment method names
    GENERALIZED_ADJUSTMENT_DEFAULT = "default"
    # Not supported yet
    GENERALIZED_ADJUSTMENT_EXHAUSTIVE = "exhaustive-search"


MAX_BACKDOOR_ITERATIONS = 100000

METHOD_NAMES = {
    BackdoorAdjustment.BACKDOOR_DEFAULT,
    BackdoorAdjustment.BACKDOOR_EXHAUSTIVE,
    BackdoorAdjustment.BACKDOOR_MIN,
    BackdoorAdjustment.BACKDOOR_MAX,
    BackdoorAdjustment.BACKDOOR_EFFICIENT,
    BackdoorAdjustment.BACKDOOR_MIN_EFFICIENT,
    BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
}
EFFICIENT_METHODS = {
    BackdoorAdjustment.BACKDOOR_EFFICIENT,
    BackdoorAdjustment.BACKDOOR_MIN_EFFICIENT,
    BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
}
DEFAULT_BACKDOOR_METHOD = BackdoorAdjustment.BACKDOOR_DEFAULT


class AutoIdentifier:
    """Class that implements different identification methods.

    Currently supports backdoor, general adjustment, and instrumental variable identification methods. The identification is based on the causal graph provided.

    This class is for backwards compatibility with CausalModel
    Will be deprecated in the future in favor of function call auto_identify_effect()

    """

    def __init__(
        self,
        estimand_type: EstimandType,
        backdoor_adjustment: BackdoorAdjustment = BackdoorAdjustment.BACKDOOR_DEFAULT,
        optimize_backdoor: bool = False,
        costs: Optional[List] = None,
        # By default, we will just compute a minimal adjustment set
        generalized_adjustment: GeneralizedAdjustment = GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT,
    ):
        self.estimand_type = estimand_type
        self.backdoor_adjustment = backdoor_adjustment
        self.optimize_backdoor = optimize_backdoor
        self.costs = costs
        self.generalized_adjustment = generalized_adjustment
        self.logger = logging.getLogger(__name__)

    def identify_effect(
        self,
        graph: nx.DiGraph,
        action_nodes: Union[str, List[str]],
        outcome_nodes: Union[str, List[str]],
        observed_nodes: Union[str, List[str]],
        conditional_node_names: List[str] = None,
    ):
        estimand = identify_effect_auto(
            graph,
            action_nodes,
            outcome_nodes,
            observed_nodes,
            self.estimand_type,
            conditional_node_names,
            self.backdoor_adjustment,
            self.optimize_backdoor,
            self.costs,
            self.generalized_adjustment,
        )

        estimand.identifier = self

        return estimand

    def identify_backdoor(
        self,
        graph: nx.DiGraph,
        action_nodes: List[str],
        outcome_nodes: List[str],
        observed_nodes: List[str],
        include_unobserved: bool = False,
        dseparation_algo: str = "default",
        direct_effect: bool = False,
    ):
        return identify_backdoor(
            graph,
            action_nodes,
            outcome_nodes,
            observed_nodes,
            self.backdoor_adjustment,
            include_unobserved,
            dseparation_algo,
            direct_effect,
        )


def identify_effect_auto(
    graph: nx.DiGraph,
    action_nodes: Union[str, List[str]],
    outcome_nodes: Union[str, List[str]],
    observed_nodes: Union[str, List[str]],
    estimand_type: EstimandType,
    conditional_node_names: List[str] = None,
    backdoor_adjustment: BackdoorAdjustment = BackdoorAdjustment.BACKDOOR_DEFAULT,
    optimize_backdoor: bool = False,
    costs: Optional[List] = None,
    generalized_adjustment: GeneralizedAdjustment = GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT,
) -> IdentifiedEstimand:
    """Main method that returns an identified estimand (if one exists).

    If estimand_type is non-parametric ATE, then  uses backdoor, instrumental variable and frontdoor identification methods,  to check if an identified estimand exists, based on the causal graph.

    :param optimize_backdoor: if True, uses an optimised algorithm to compute the backdoor sets
    :param costs: non-negative costs associated with variables in the graph. Only used
    for estimand_type='non-parametric-ate' and backdoor_adjustment='efficient-mincost-adjustment'. If
    no costs are provided by the user, and backdoor_adjustment='efficient-mincost-adjustment', costs
    are assumed to be equal to one for all variables in the graph.
    :param conditional_node_names: variables that are used to determine treatment. If none are
    provided, it is assumed that the intervention is static.
    :param generalized_adjustment: specify whether to return a single minimal adjustment set which
    matches the general adjustment criterion ("default"), or to exhaustively compute all such adjustment sets ("exhaustive-search"). For now
    only minimal adjustment sets are supported.
    :returns:  target estimand, an instance of the IdentifiedEstimand class
    """

    observed_nodes = parse_state(observed_nodes)
    action_nodes = parse_state(action_nodes)
    outcome_nodes = parse_state(outcome_nodes)

    # First, check if there is a directed path from action to outcome
    if not has_directed_path(graph, action_nodes, outcome_nodes):
        logger.warning("No directed path from treatment to outcome. Causal Effect is zero.")
        return IdentifiedEstimand(
            None,
            treatment_variable=action_nodes,
            outcome_variable=outcome_nodes,
            no_directed_path=True,
        )
    if estimand_type == EstimandType.NONPARAMETRIC_ATE:
        return identify_ate_effect(
            graph,
            action_nodes,
            outcome_nodes,
            observed_nodes,
            backdoor_adjustment,
            optimize_backdoor,
            estimand_type,
            costs,
            conditional_node_names,
            generalized_adjustment,
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_NDE:
        return identify_nde_effect(
            graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment, estimand_type
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_NIE:
        return identify_nie_effect(
            graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment, estimand_type
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_CDE:
        return identify_cde_effect(
            graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment, estimand_type
        )
    else:
        raise ValueError(
            "Estimand type is not supported. Use either {0}, {1}, or {2}.".format(
                EstimandType.NONPARAMETRIC_ATE,
                EstimandType.NONPARAMETRIC_CDE,
                EstimandType.NONPARAMETRIC_NDE,
                EstimandType.NONPARAMETRIC_NIE,
            )
        )


def identify_ate_effect(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    optimize_backdoor: bool,
    estimand_type: EstimandType,
    costs: List,
    conditional_node_names: List[str] = None,
    generalized_adjustment: GeneralizedAdjustment = GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT,
):
    estimands_dict = {}
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    ### 1. BACKDOOR IDENTIFICATION
    # Pick algorithm to compute backdoor sets according to method chosen
    if backdoor_adjustment not in EFFICIENT_METHODS:
        # First, checking if there are any valid backdoor adjustment sets
        if optimize_backdoor == False:
            backdoor_sets = identify_backdoor(graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment)
        else:
            from dowhy.causal_identifier.backdoor import Backdoor

            path = Backdoor(graph, action_nodes, outcome_nodes)
            backdoor_sets = path.get_backdoor_vars()
    elif backdoor_adjustment in EFFICIENT_METHODS:
        backdoor_sets = identify_efficient_backdoor(
            graph,
            action_nodes,
            outcome_nodes,
            observed_nodes,
            backdoor_adjustment,
            costs,
            conditional_node_names=conditional_node_names,
        )
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        action_nodes, outcome_nodes, observed_nodes, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    if len(backdoor_variables_dict) > 0:
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    else:
        estimands_dict["backdoor"] = None
    ### 2. INSTRUMENTAL VARIABLE IDENTIFICATION
    # Now checking if there is also a valid iv estimand
    instrument_names = get_instruments(graph, action_nodes, outcome_nodes)
    logger.info("Instrumental variables for treatment and outcome:" + str(instrument_names))
    if len(instrument_names) > 0:
        iv_estimand_expr = construct_iv_estimand(
            action_nodes,
            outcome_nodes,
            instrument_names,
        )
        logger.debug("Identified expression = " + str(iv_estimand_expr))
        estimands_dict["iv"] = iv_estimand_expr
    else:
        estimands_dict["iv"] = None

    ### 3. FRONTDOOR IDENTIFICATION
    # Now checking if there is a valid frontdoor variable
    frontdoor_variables_names = identify_frontdoor(graph, action_nodes, outcome_nodes, observed_nodes)
    logger.info("Frontdoor variables for treatment and outcome:" + str(frontdoor_variables_names))
    if len(frontdoor_variables_names) > 0:
        frontdoor_estimand_expr = construct_frontdoor_estimand(
            action_nodes,
            outcome_nodes,
            frontdoor_variables_names,
        )
        logger.debug("Identified expression = " + str(frontdoor_estimand_expr))
        estimands_dict["frontdoor"] = frontdoor_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, action_nodes, outcome_nodes, frontdoor_variables_names, observed_nodes, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, action_nodes, frontdoor_variables_names, outcome_nodes, observed_nodes, backdoor_adjustment
        )
    else:
        estimands_dict["frontdoor"] = None

    ### 4. GENERAL ADJUSTMENT IDENTIFICATION
    # This generalizes the backdoor criterion, identifying other valid covariate adjustment sets that might not
    # satisfy the backdoor criterion. This capability requires python >=3.10
    adjustment_variables_dict = default_adjustment_id = None
    if sys.version_info >= (3, 10):
        adjustment_sets = identify_generalized_adjustment_set(
            graph, action_nodes, outcome_nodes, observed_nodes, generalized_adjustment
        )
        logger.info("Number of general adjustment sets found: " + str(len(adjustment_sets)))
        estimands_dict, adjustment_variables_dict = build_adjustment_set_estimands_dict(
            action_nodes, outcome_nodes, observed_nodes, adjustment_sets, estimands_dict
        )
        default_adjustment_id = get_default_adjustment_set_id(
            graph, action_nodes, outcome_nodes, adjustment_variables_dict
        )
        if len(adjustment_variables_dict) > 0:
            estimands_dict["general_adjustment"] = estimands_dict.get(str(default_adjustment_id), None)
            adjustment_variables_dict["general_adjustment"] = adjustment_variables_dict.get(
                str(default_adjustment_id), None
            )
        else:
            estimands_dict["general_adjustment"] = None
    else:
        logger.warning(
            f"Generalized covariate adjustment identification is not supported for the detected Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
        )
    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=action_nodes,
        outcome_variable=outcome_nodes,
        estimand_type=estimand_type,
        estimands=estimands_dict,
        backdoor_variables=backdoor_variables_dict,
        general_adjustment_variables=adjustment_variables_dict,
        instrumental_variables=instrument_names,
        frontdoor_variables=frontdoor_variables_names,
        mediation_first_stage_confounders=mediation_first_stage_confounders,
        mediation_second_stage_confounders=mediation_second_stage_confounders,
        default_backdoor_id=default_backdoor_id,
        default_adjustment_set_id=default_adjustment_id,
    )
    return estimand


def identify_cde_effect(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
):
    """Identify controlled direct effect. For a definition, see Vanderwheele (2011).
    Controlled direct and mediated effects: definition, identification and bounds.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4193506/

    Using do-calculus rules, identification yields a adjustment set.
    It is based on the principle that under a graph where the direct edge from treatment
    to outcome is removed, conditioning on the adjustment set should d-separate
    treatment and outcome.
    """
    estimands_dict = {}
    # Pick algorithm to compute backdoor sets according to method chosen
    backdoor_sets = identify_backdoor(
        graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment, direct_effect=True
    )
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        action_nodes, outcome_nodes, observed_nodes, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    if len(backdoor_variables_dict) > 0:
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    else:
        estimands_dict["backdoor"] = None

    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=action_nodes,
        outcome_variable=outcome_nodes,
        estimand_type=estimand_type,
        estimands=estimands_dict,
        backdoor_variables=backdoor_variables_dict,
        instrumental_variables=None,
        frontdoor_variables=None,
        mediation_first_stage_confounders=None,
        mediation_second_stage_confounders=None,
        default_backdoor_id=default_backdoor_id,
    )
    return estimand


def identify_nie_effect(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
):
    estimands_dict = {}
    ### 1. FIRST DOING BACKDOOR IDENTIFICATION
    # First, checking if there are any valid backdoor adjustment sets
    backdoor_sets = identify_backdoor(graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        action_nodes, outcome_nodes, observed_nodes, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

    ### 2. SECOND, CHECKING FOR MEDIATORS
    # Now checking if there are valid mediator variables
    estimands_dict = {}  # Need to reinitialize this dictionary to avoid including the backdoor sets
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    mediators_names = identify_mediation(graph, action_nodes, outcome_nodes)
    logger.info("Mediators for treatment and outcome:" + str(mediators_names))
    if len(mediators_names) > 0:
        mediation_estimand_expr = construct_mediation_estimand(
            estimand_type,
            action_nodes,
            outcome_nodes,
            mediators_names,
        )
        logger.debug("Identified expression = " + str(mediation_estimand_expr))
        estimands_dict["mediation"] = mediation_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, action_nodes, outcome_nodes, mediators_names, observed_nodes, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, action_nodes, mediators_names, outcome_nodes, observed_nodes, backdoor_adjustment
        )
    else:
        estimands_dict["mediation"] = None
    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=action_nodes,
        outcome_variable=outcome_nodes,
        estimand_type=estimand_type,
        estimands=estimands_dict,
        backdoor_variables=backdoor_variables_dict,
        instrumental_variables=None,
        frontdoor_variables=None,
        mediator_variables=mediators_names,
        mediation_first_stage_confounders=mediation_first_stage_confounders,
        mediation_second_stage_confounders=mediation_second_stage_confounders,
        default_backdoor_id=None,
    )
    return estimand


def identify_nde_effect(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
):
    estimands_dict = {}
    ### 1. FIRST DOING BACKDOOR IDENTIFICATION
    # First, checking if there are any valid backdoor adjustment sets
    backdoor_sets = identify_backdoor(graph, action_nodes, outcome_nodes, observed_nodes, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        action_nodes, outcome_nodes, observed_nodes, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

    ### 2. SECOND, CHECKING FOR MEDIATORS
    # Now checking if there are valid mediator variables
    estimands_dict = {}
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    mediators_names = identify_mediation(graph, action_nodes, outcome_nodes)
    logger.info("Mediators for treatment and outcome:" + str(mediators_names))
    if len(mediators_names) > 0:
        mediation_estimand_expr = construct_mediation_estimand(
            estimand_type,
            action_nodes,
            outcome_nodes,
            mediators_names,
        )
        logger.debug("Identified expression = " + str(mediation_estimand_expr))
        estimands_dict["mediation"] = mediation_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, action_nodes, outcome_nodes, mediators_names, observed_nodes, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, action_nodes, mediators_names, outcome_nodes, observed_nodes, backdoor_adjustment
        )
    else:
        estimands_dict["mediation"] = None
    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=action_nodes,
        outcome_variable=outcome_nodes,
        estimand_type=estimand_type,
        estimands=estimands_dict,
        backdoor_variables=backdoor_variables_dict,
        instrumental_variables=None,
        frontdoor_variables=None,
        mediator_variables=mediators_names,
        mediation_first_stage_confounders=mediation_first_stage_confounders,
        mediation_second_stage_confounders=mediation_second_stage_confounders,
        default_backdoor_id=None,
    )
    return estimand


def identify_backdoor(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    include_unobserved: bool = False,
    dseparation_algo: str = "default",
    direct_effect: bool = False,
) -> List[AdjustmentSet]:
    backdoor_sets = []
    backdoor_paths = None
    bdoor_graph = None
    observed_nodes = set(observed_nodes)
    if dseparation_algo == "naive":
        backdoor_paths = get_backdoor_paths(graph, action_nodes, outcome_nodes)
    elif dseparation_algo == "default":
        bdoor_graph = do_surgery(
            graph,
            action_nodes,
            target_node_names=outcome_nodes,
            remove_outgoing_edges=True,
            remove_only_direct_edges_to_target=direct_effect,
        )
    else:
        raise ValueError(f"d-separation algorithm {dseparation_algo} is not supported")
    backdoor_adjustment = (
        backdoor_adjustment if backdoor_adjustment != BackdoorAdjustment.BACKDOOR_DEFAULT else DEFAULT_BACKDOOR_METHOD
    )

    # First, checking if empty set is a valid backdoor set
    empty_set = set()
    check = check_valid_backdoor_set(
        graph,
        action_nodes,
        outcome_nodes,
        empty_set,
        backdoor_paths=backdoor_paths,
        new_graph=bdoor_graph,
        dseparation_algo=dseparation_algo,
    )
    if check["is_dseparated"]:
        backdoor_sets.append(AdjustmentSet(AdjustmentSet.BACKDOOR, empty_set))
        # If the method is `minimal-adjustment`, return the empty set right away.
        if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN:
            return backdoor_sets

    # Second, checking for all other sets of variables. If include_unobserved is false, then only observed variables are eligible.
    eligible_variables = (
        set([node for node in graph.nodes if include_unobserved or node in observed_nodes])
        - set(action_nodes)
        - set(outcome_nodes)
    )

    if direct_effect:
        # only remove descendants of Y
        # also allow any causes of Y that are not caused by T (for lower variance)
        eligible_variables -= get_descendants(graph, outcome_nodes)
    else:
        # remove descendants of T (mediators) and descendants of Y
        eligible_variables -= get_descendants(graph, action_nodes)
    # If var is d-separated from both treatment or outcome, it cannot
    # be a part of the backdoor set
    filt_eligible_variables = set()
    for var in eligible_variables:
        dsep_treat_var = check_dseparation(graph, action_nodes, parse_state(var), set())
        dsep_outcome_var = check_dseparation(graph, outcome_nodes, parse_state(var), set())
        if not dsep_outcome_var or not dsep_treat_var:
            filt_eligible_variables.add(var)
    if backdoor_adjustment in METHOD_NAMES:
        backdoor_sets, found_valid_adjustment_set = find_valid_adjustment_sets(
            graph,
            action_nodes,
            outcome_nodes,
            observed_nodes,
            backdoor_paths,
            bdoor_graph,
            dseparation_algo,
            backdoor_sets,
            filt_eligible_variables,
            backdoor_adjustment=backdoor_adjustment,
            max_iterations=MAX_BACKDOOR_ITERATIONS,
        )
        if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_DEFAULT and found_valid_adjustment_set:
            # repeat the above search with BACKDOOR_MIN
            backdoor_sets, _ = find_valid_adjustment_sets(
                graph,
                action_nodes,
                outcome_nodes,
                observed_nodes,
                backdoor_paths,
                bdoor_graph,
                dseparation_algo,
                backdoor_sets,
                filt_eligible_variables,
                backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MIN,
                max_iterations=MAX_BACKDOOR_ITERATIONS,
            )
    else:
        raise ValueError(
            f"Identifier method {backdoor_adjustment} not supported. Try one of the following: {METHOD_NAMES}"
        )
    return backdoor_sets


def identify_efficient_backdoor(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
    costs: List,
    conditional_node_names: List[str] = None,
):
    """Method implementing algorithms to compute efficient backdoor sets, as
    described in Rotnitzky and Smucler (2020), Smucler, Sapienza and Rotnitzky (2021)
    and Smucler and Rotnitzky (2022).

    For backdoor_adjustment='efficient-adjustment', computes an optimal backdoor set,
    that is, a backdoor set comprised of observable variables that yields non-parametric
    estimators of the interventional mean with the smallest asymptotic variance
    among those that are based on observable backdoor sets. This optimal backdoor
    set always exists when no variables are latent, and the algorithm is guaranteed to compute
    it in this case. Under a non-parametric graphical model with latent variables,
    such a backdoor set can fail to exist. When certain sufficient conditions under which it is
    known that such a backdoor set exists are not satisfied, an error is raised.

    For backdoor_adjustment='efficient-minimal-adjustment', computes an optimal minimal backdoor set,
    that is, a minimal backdoor set comprised of observable variables that yields non-parametric
    estimators of the interventional mean with the smallest asymptotic variance
    among those that are based on observable minimal backdoor sets.

    For backdoor_adjustment='efficient-mincost-adjustment', computes an optimal minimum cost backdoor set,
    that is, a minimum cost backdoor set comprised of observable variables that yields non-parametric
    estimators of the interventional mean with the smallest asymptotic variance
    among those that are based on observable minimum cost backdoor sets. The cost
    of a backdoor set is defined as the sum of the costs of the variables that comprise it.

    The various optimal backdoor sets computed by this method are not only optimal under
    non-parametric graphical models and non-parametric estimators of interventional mean,
    but also under linear graphical models and OLS estimators, per results in Henckel, Perkovic
    and Maathuis (2020).

    :param costs: a list with non-negative costs associated with variables in the graph. Only used
    for estimatand_type='non-parametric-ate' and backdoor_adjustment='efficient-mincost-adjustment'. If
    not costs are provided by the user, and backdoor_adjustment='efficient-mincost-adjustment', costs
    are assumed to be equal to one for all variables in the graph. The structure of the list should
    be of the form [(node, {"cost": x}) for node in nodes].
    :param conditional_node_names: variables that are used to determine treatment. If none are
    provided, it is assumed that the intervention sets the treatment to a constant.
    :returns:  backdoor_sets, a list of dictionaries, with each dictionary
    having as values a backdoor set.
    """
    if costs is None and backdoor_adjustment == "efficient-mincost-adjustment":
        logger.warning("No costs were passed, so they will be assumed to be constant and equal to 1.")
    efficient_bd = EfficientBackdoor(
        graph=graph,
        action_nodes=action_nodes,
        outcome_nodes=outcome_nodes,
        observed_nodes=observed_nodes,
        conditional_node_names=conditional_node_names,
        costs=costs,
    )
    if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_EFFICIENT:
        backdoor_set = efficient_bd.optimal_adj_set()
        backdoor_sets = [AdjustmentSet(AdjustmentSet.BACKDOOR, tuple(backdoor_set))]
    elif backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN_EFFICIENT:
        backdoor_set = efficient_bd.optimal_minimal_adj_set()
        backdoor_sets = [AdjustmentSet(AdjustmentSet.BACKDOOR, tuple(backdoor_set))]
    elif backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT:
        backdoor_set = efficient_bd.optimal_mincost_adj_set()
        backdoor_sets = [AdjustmentSet(AdjustmentSet.BACKDOOR, tuple(backdoor_set))]
    return backdoor_sets


def find_valid_adjustment_sets(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_paths: List,
    bdoor_graph: nx.DiGraph,
    dseparation_algo: str,
    backdoor_sets: List,
    filt_eligible_variables: List,
    backdoor_adjustment: BackdoorAdjustment,
    max_iterations: int,
):
    num_iterations = 0
    found_valid_adjustment_set = False
    is_all_observed = set(graph.nodes) == set(observed_nodes)
    # If `minimal-adjustment` method is specified, start the search from the set with minimum size. Otherwise, start from the largest.
    set_sizes = (
        range(1, len(filt_eligible_variables) + 1, 1)
        if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN
        else range(len(filt_eligible_variables), 0, -1)
    )
    for size_candidate_set in set_sizes:
        for candidate_set in itertools.combinations(filt_eligible_variables, size_candidate_set):
            check = check_valid_backdoor_set(
                graph,
                action_nodes,
                outcome_nodes,
                candidate_set,
                backdoor_paths=backdoor_paths,
                new_graph=bdoor_graph,
                dseparation_algo=dseparation_algo,
            )
            logger.debug(
                "Candidate backdoor set: {0}, is_dseparated: {1}".format(candidate_set, check["is_dseparated"])
            )
            if check["is_dseparated"]:
                backdoor_sets.append(AdjustmentSet(AdjustmentSet.BACKDOOR, candidate_set))
                found_valid_adjustment_set = True
            num_iterations += 1
            if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_EXHAUSTIVE and num_iterations > max_iterations:
                logger.warning(f"Max number of iterations {max_iterations} reached.")
                break
        # If the backdoor method is `maximal-adjustment` or `minimal-adjustment`, return the first found adjustment set.
        if (
            backdoor_adjustment
            in {
                BackdoorAdjustment.BACKDOOR_DEFAULT,
                BackdoorAdjustment.BACKDOOR_MAX,
                BackdoorAdjustment.BACKDOOR_MIN,
            }
            and found_valid_adjustment_set
        ):
            break
        # If all variables are observed, and the biggest eligible set
        # does not satisfy backdoor, then none of its subsets will.
        if (
            backdoor_adjustment in {BackdoorAdjustment.BACKDOOR_DEFAULT, BackdoorAdjustment.BACKDOOR_MAX}
            and is_all_observed
        ):
            break
        if num_iterations > max_iterations:
            logger.warning(f"Max number of iterations {max_iterations} reached. Could not find a valid backdoor set.")
            break
    return backdoor_sets, found_valid_adjustment_set


def get_default_adjustment_set_id(
    graph: nx.DiGraph, action_nodes: List[str], outcome_nodes: List[str], adjustment_sets_dict: Dict
):
    # Adding a None estimand if no adjustment set found
    if len(adjustment_sets_dict) == 0:
        return None

    # Default set contains minimum possible number of instrumental variables, to prevent lowering variance in the treatment variable.
    instrument_names = set(get_instruments(graph, action_nodes, outcome_nodes))
    iv_count_dict = {
        key: len(set(adjustment_set).intersection(instrument_names))
        for key, adjustment_set in adjustment_sets_dict.items()
    }
    min_iv_count = min(iv_count_dict.values())
    min_iv_keys = {key for key, iv_count in iv_count_dict.items() if iv_count == min_iv_count}
    min_iv_adjustment_sets_dict = {key: adjustment_sets_dict[key] for key in min_iv_keys}

    # Default set is the one with the least number of adjustment variables (optimizing for efficiency)
    min_set_length = 1000000
    default_key = None
    for key, adjustment_set in min_iv_adjustment_sets_dict.items():
        if len(adjustment_set) < min_set_length:
            min_set_length = len(adjustment_set)
            default_key = key
    return default_key


def build_adjustment_set_estimands_dict(
    treatment_names: List[str],
    outcome_names: List[str],
    observed_nodes: List[str],
    adjustment_sets: List[AdjustmentSet],
    estimands_dict: Dict,
):
    """Build the final dict for adjustment sets by filtering unobserved variables if needed."""
    adjustment_variables_dict = {}
    observed_nodes = set(observed_nodes)
    is_identified = [set(aset.get_adjustment_variables()).issubset(observed_nodes) for aset in adjustment_sets]

    if any(is_identified):
        logger.info("Causal effect can be identified.")
        adjustment_sets_filtered = [
            aset for aset in adjustment_sets if set(aset.get_adjustment_variables()).issubset(observed_nodes)
        ]
    else:  # there is unobserved confounding
        logger.warning("Adjustment set identification failed.")
        adjustment_sets_filtered = []

    for i, adjSet in enumerate(adjustment_sets_filtered):
        adjustment_estimand_expr = construct_adjustment_estimand(
            treatment_names, outcome_names, adjSet.get_adjustment_variables()
        )
        logger.debug("Identified expression = " + str(adjustment_estimand_expr))
        estimands_dict[adjSet.get_adjustment_type() + str(i + 1)] = adjustment_estimand_expr
        adjustment_variables_dict[adjSet.get_adjustment_type() + str(i + 1)] = list(adjSet.get_adjustment_variables())
    return estimands_dict, adjustment_variables_dict


def identify_frontdoor(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    dseparation_algo: str = "default",
):
    """Find a valid frontdoor variable set if it exists."""
    frontdoor_var = None
    frontdoor_paths = None
    fdoor_graph = None
    if dseparation_algo == "default":
        cond1_graph = do_surgery(graph, action_nodes, remove_incoming_edges=True)
    elif dseparation_algo == "naive":
        frontdoor_paths = get_all_directed_paths(graph, action_nodes, outcome_nodes)
    else:
        raise ValueError(f"d-separation algorithm {dseparation_algo} is not supported")

    eligible_variables = (
        get_descendants(graph, action_nodes)
        - set(action_nodes)
        - set(outcome_nodes)
        - set(get_descendants(graph, outcome_nodes))
    )
    eligible_variables = eligible_variables.intersection(set(observed_nodes))
    set_sizes = range(1, len(eligible_variables) + 1, 1)
    for size_candidate_set in set_sizes:
        for candidate_set in itertools.combinations(eligible_variables, size_candidate_set):
            candidate_set = list(candidate_set)
            # Cond 1: All directed paths intercepted by candidate_var
            cond1 = check_valid_frontdoor_set(
                graph,
                action_nodes,
                outcome_nodes,
                candidate_set,
                frontdoor_paths=frontdoor_paths,
                new_graph=cond1_graph,
                dseparation_algo=dseparation_algo,
            )
            logger.debug("Candidate frontdoor set: {0}, Cond1: is_dseparated: {1}".format(candidate_set, cond1))
            if not cond1:
                continue
            # Cond 2: No confounding between treatment and candidate var
            cond2 = check_valid_backdoor_set(
                graph,
                action_nodes,
                candidate_set,
                set(),
                backdoor_paths=None,
                dseparation_algo=dseparation_algo,
            )["is_dseparated"]
            if not cond2:
                continue
            # Cond 3: treatment blocks all confounding between candidate_var and outcome
            bdoor_graph2 = do_surgery(graph, candidate_set, remove_outgoing_edges=True)
            cond3 = check_valid_backdoor_set(
                graph,
                candidate_set,
                outcome_nodes,
                action_nodes,
                backdoor_paths=None,
                new_graph=bdoor_graph2,
                dseparation_algo=dseparation_algo,
            )["is_dseparated"]
            is_valid_frontdoor = cond1 and cond2 and cond3
            if is_valid_frontdoor:
                frontdoor_var = candidate_set
                break
    return parse_state(frontdoor_var)


def identify_generalized_adjustment_set(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    generalized_adjustment: GeneralizedAdjustment = GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT,
) -> List[AdjustmentSet]:
    """Find an adjustment set if one exists. This generalizes the backdoor criterion, using a complete criterion
    which guarantees an adjustment set will be found if one exists.

    Currently only supports returning a single minimal adjustment set.

    References
    ----------
    [1] Benito van der Zander, Maciej Liśkiewicz, and Johannes Textor. "Constructing Separators and
       Adjustment Sets in Ancestral Graphs." In Proceedings of UAI 2014, pages 907–916,
       2014.
    """

    graph_pbd = get_proper_backdoor_graph(graph, action_nodes, outcome_nodes)
    pcp_nodes = get_proper_causal_path_nodes(graph, action_nodes, outcome_nodes)
    dpcp_nodes = get_descendants(graph, pcp_nodes).union(pcp_nodes)

    if generalized_adjustment == GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT:
        # In default case, we don't exhaustively find all adjustment sets
        adjustment_set = nx.algorithms.find_minimal_d_separator(
            graph_pbd,
            set(action_nodes),
            set(outcome_nodes),
            # Require the adjustment set to consist only of observed nodes
            restricted=((set(graph_pbd.nodes) - set(dpcp_nodes)) & set(observed_nodes)),
        )
        if adjustment_set is None:
            logger.info("No adjustment sets found.")
            return []
        return [AdjustmentSet(AdjustmentSet.GENERAL, adjustment_set)]
    elif generalized_adjustment == GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_EXHAUSTIVE:
        raise ValueError("Exhaustive identification of general adjustment sets is not yet supported.")
    else:
        raise ValueError("Please provide a valid type of Generalized Adjustment")


def identify_mediation(graph: nx.DiGraph, action_nodes: List[str], outcome_nodes: List[str]):
    """Find a valid mediator if it exists.

    Currently only supports a single variable mediator set.
    """
    mediation_var = None
    mediation_paths = get_all_directed_paths(graph, action_nodes, outcome_nodes)
    eligible_variables = get_descendants(graph, action_nodes) - set(outcome_nodes)
    # For simplicity, assuming a one-variable mediation set
    for candidate_var in eligible_variables:
        is_valid_mediation = check_valid_mediation_set(
            graph,
            action_nodes,
            outcome_nodes,
            parse_state(candidate_var),
            mediation_paths=mediation_paths,
        )
        logger.debug("Candidate mediation set: {0}, on_mediating_path: {1}".format(candidate_var, is_valid_mediation))
        if is_valid_mediation:
            mediation_var = candidate_var
            break
    return parse_state(mediation_var)


def identify_mediation_first_stage_confounders(
    graph: nx.DiGraph,
    action_nodes: List[str],
    outcome_nodes: List[str],
    mediator_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
):
    # Create estimands dict as per the API for backdoor, but do not return it
    estimands_dict = {}
    backdoor_sets = identify_backdoor(graph, action_nodes, mediator_nodes, observed_nodes, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        action_nodes,
        mediator_nodes,
        observed_nodes,
        backdoor_sets,
        estimands_dict,
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    return backdoor_variables_dict


def identify_mediation_second_stage_confounders(
    graph: nx.DiGraph,
    action_nodes: List[str],
    mediator_nodes: List[str],
    outcome_nodes: List[str],
    observed_nodes: List[str],
    backdoor_adjustment: BackdoorAdjustment,
):
    # Create estimands dict as per the API for backdoor, but do not return it
    estimands_dict = {}
    backdoor_sets = identify_backdoor(graph, mediator_nodes, outcome_nodes, observed_nodes, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_adjustment_set_estimands_dict(
        mediator_nodes,
        outcome_nodes,
        observed_nodes,
        backdoor_sets,
        estimands_dict,
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_adjustment_set_id(graph, action_nodes, outcome_nodes, backdoor_variables_dict)
    estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    return backdoor_variables_dict


def construct_adjustment_estimand(treatment_name: List[str], outcome_name: List[str], common_causes: List[str]):
    # TODO: outputs string for now, but ideally should do symbolic
    # expressions Mon 19 Feb 2018 04:54:17 PM DST
    # TODO Better support for multivariate treatments

    expr = None
    outcome_name = outcome_name[0]
    num_expr_str = outcome_name
    if len(common_causes) > 0:
        num_expr_str += "|" + ",".join(common_causes)
    expr = "d(" + num_expr_str + ")/d" + ",".join(treatment_name)
    sym_mu = sp.Symbol("mu")
    sym_sigma = sp.Symbol("sigma", positive=True)
    sym_outcome = spstats.Normal(num_expr_str, sym_mu, sym_sigma)
    sym_treatment_symbols = [sp.Symbol(t) for t in treatment_name]
    sym_treatment = sp.Array(sym_treatment_symbols)
    sym_conditional_outcome = spstats.Expectation(sym_outcome)
    sym_effect = sp.Derivative(sym_conditional_outcome, sym_treatment)

    sym_assumptions = {
        "Unconfoundedness": (
            "If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{1}" " then P({1}|{0},{2},U) = P({1}|{0},{2})"
        ).format(",".join(treatment_name), outcome_name, ",".join(common_causes))
    }

    estimand = {"estimand": sym_effect, "assumptions": sym_assumptions}
    return estimand


def construct_iv_estimand(treatment_name: List[str], outcome_name: List[str], instrument_names: List[str]):
    # TODO: support multivariate treatments better.
    expr = None
    outcome_name = outcome_name[0]
    sym_outcome = spstats.Normal(outcome_name, 0, 1)
    sym_treatment_symbols = [spstats.Normal(t, 0, 1) for t in treatment_name]
    sym_treatment = sp.Array(sym_treatment_symbols)
    sym_instrument_symbols = [sp.Symbol(inst) for inst in instrument_names]
    sym_instrument = sp.Array(sym_instrument_symbols)  # ",".join(instrument_names))
    sym_outcome_derivative = sp.Derivative(sym_outcome, sym_instrument)
    sym_treatment_derivative = sp.Derivative(sym_treatment, sym_instrument)
    sym_effect = spstats.Expectation(sym_outcome_derivative / sym_treatment_derivative)
    sym_assumptions = {
        "As-if-random": (
            "If U\N{RIGHTWARDS ARROW}\N{RIGHTWARDS ARROW}{0} then "
            "\N{NOT SIGN}(U \N{RIGHTWARDS ARROW}\N{RIGHTWARDS ARROW}{{{1}}})"
        ).format(outcome_name, ",".join(instrument_names)),
        "Exclusion": (
            "If we remove {{{0}}}\N{RIGHTWARDS ARROW}{{{1}}}, then " "\N{NOT SIGN}({{{0}}}\N{RIGHTWARDS ARROW}{2})"
        ).format(",".join(instrument_names), ",".join(treatment_name), outcome_name),
    }

    estimand = {"estimand": sym_effect, "assumptions": sym_assumptions}
    return estimand


def construct_frontdoor_estimand(
    treatment_name: List[str], outcome_name: List[str], frontdoor_variables_names: List[str]
):
    # TODO: support multivariate treatments better.
    expr = None
    outcome_name = outcome_name[0]
    sym_outcome = spstats.Normal(outcome_name, 0, 1)
    sym_treatment_symbols = [spstats.Normal(t, 0, 1) for t in treatment_name]
    sym_treatment = sp.Array(sym_treatment_symbols)
    sym_frontdoor_symbols = [sp.Symbol(inst) for inst in frontdoor_variables_names]
    sym_frontdoor = sp.Array(sym_frontdoor_symbols)  # ",".join(instrument_names))
    sym_outcome_derivative = sp.Derivative(sym_outcome, sym_frontdoor)
    sym_treatment_derivative = sp.Derivative(sym_frontdoor, sym_treatment)
    sym_effect = spstats.Expectation(sym_treatment_derivative * sym_outcome_derivative)
    sym_assumptions = {
        "Full-mediation": ("{2} intercepts (blocks) all directed paths from {0} to {1}.").format(
            ",".join(treatment_name),
            ",".join(outcome_name),
            ",".join(frontdoor_variables_names),
        ),
        "First-stage-unconfoundedness": (
            "If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}" " then P({1}|{0},U) = P({1}|{0})"
        ).format(",".join(treatment_name), ",".join(frontdoor_variables_names)),
        "Second-stage-unconfoundedness": (
            "If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}" " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
        ).format(
            ",".join(treatment_name),
            outcome_name,
            ",".join(frontdoor_variables_names),
        ),
    }

    estimand = {"estimand": sym_effect, "assumptions": sym_assumptions}
    return estimand


def construct_mediation_estimand(
    estimand_type: EstimandType, action_nodes: List[str], outcome_nodes: List[str], mediator_nodes: List[str]
):
    # TODO: support multivariate treatments better.
    expr = None
    if estimand_type in (
        EstimandType.NONPARAMETRIC_NDE,
        EstimandType.NONPARAMETRIC_NIE,
    ):
        outcome_nodes = outcome_nodes[0]
        sym_outcome = spstats.Normal(outcome_nodes, 0, 1)
        sym_treatment_symbols = [spstats.Normal(t, 0, 1) for t in action_nodes]
        sym_treatment = sp.Array(sym_treatment_symbols)
        sym_mediators_symbols = [sp.Symbol(inst) for inst in mediator_nodes]
        sym_mediators = sp.Array(sym_mediators_symbols)
        sym_outcome_derivative = sp.Derivative(sym_outcome, sym_mediators)
        sym_treatment_derivative = sp.Derivative(sym_mediators, sym_treatment)
        # For direct effect
        num_expr_str = outcome_nodes
        if len(mediator_nodes) > 0:
            num_expr_str += "|" + ",".join(mediator_nodes)
        sym_mu = sp.Symbol("mu")
        sym_sigma = sp.Symbol("sigma", positive=True)
        sym_conditional_outcome = spstats.Normal(num_expr_str, sym_mu, sym_sigma)
        sym_directeffect_derivative = sp.Derivative(sym_conditional_outcome, sym_treatment)
        if estimand_type == EstimandType.NONPARAMETRIC_NIE:
            sym_effect = spstats.Expectation(sym_treatment_derivative * sym_outcome_derivative)
        elif estimand_type == EstimandType.NONPARAMETRIC_NDE:
            sym_effect = spstats.Expectation(sym_directeffect_derivative)
        sym_assumptions = {
            "Mediation": (
                "{2} intercepts (blocks) all directed paths from {0} to {1} except the path {{{0}}}\N{RIGHTWARDS ARROW}{{{1}}}."
            ).format(
                ",".join(action_nodes),
                ",".join(outcome_nodes),
                ",".join(mediator_nodes),
            ),
            "First-stage-unconfoundedness": (
                "If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}" " then P({1}|{0},U) = P({1}|{0})"
            ).format(",".join(action_nodes), ",".join(mediator_nodes)),
            "Second-stage-unconfoundedness": (
                "If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}"
                " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
            ).format(",".join(action_nodes), outcome_nodes, ",".join(mediator_nodes)),
        }
    else:
        raise ValueError(
            "Estimand type not supported. Supported estimand types are {0} or {1}'.".format(
                EstimandType.NONPARAMETRIC_NDE,
                EstimandType.NONPARAMETRIC_NIE,
            )
        )

    estimand = {"estimand": sym_effect, "assumptions": sym_assumptions}
    return estimand
