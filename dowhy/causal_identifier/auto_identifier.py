import itertools
import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import sympy as sp
import sympy.stats as spstats

from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier.efficient_backdoor import EfficientBackdoor
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
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

    Currently supports backdoor and instrumental variable identification methods. The identification is based on the causal graph provided.

    This class is for backwards compatibility with CausalModel
    Will be deprecated in the future in favor of function call auto_identify_effect()

    """

    def __init__(
        self,
        estimand_type: EstimandType,
        backdoor_adjustment: BackdoorAdjustment = BackdoorAdjustment.BACKDOOR_DEFAULT,
        proceed_when_unidentifiable: bool = False,
        optimize_backdoor: bool = False,
        costs: Optional[List] = None,
    ):
        self.estimand_type = estimand_type
        self.backdoor_adjustment = backdoor_adjustment
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self.optimize_backdoor = optimize_backdoor
        self.costs = costs
        self.logger = logging.getLogger(__name__)

    def identify_effect(
        self,
        graph: CausalGraph,
        treatment_name: Union[str, List[str]],
        outcome_name: Union[str, List[str]],
        conditional_node_names: List[str] = None,
        **kwargs,
    ):
        estimand = identify_effect_auto(
            graph,
            treatment_name,
            outcome_name,
            self.estimand_type,
            conditional_node_names,
            self.backdoor_adjustment,
            self._proceed_when_unidentifiable,
            self.optimize_backdoor,
            self.costs,
            **kwargs,
        )

        estimand.identifier = self

        return estimand

    def identify_backdoor(
        self,
        graph: CausalGraph,
        treatment_name: List[str],
        outcome_name: str,
        include_unobserved: bool = False,
        dseparation_algo: str = "default",
        direct_effect: bool = False,
    ):
        return identify_backdoor(
            graph,
            treatment_name,
            outcome_name,
            self.backdoor_adjustment,
            include_unobserved,
            dseparation_algo,
            direct_effect,
        )


def identify_effect_auto(
    graph: CausalGraph,
    treatment_name: Union[str, List[str]],
    outcome_name: Union[str, List[str]],
    estimand_type: EstimandType,
    conditional_node_names: List[str] = None,
    backdoor_adjustment: BackdoorAdjustment = BackdoorAdjustment.BACKDOOR_DEFAULT,
    proceed_when_unidentifiable: bool = False,
    optimize_backdoor: bool = False,
    costs: Optional[List] = None,
    **kwargs,
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
    :returns:  target estimand, an instance of the IdentifiedEstimand class
    """

    treatment_name = parse_state(treatment_name)
    outcome_name = parse_state(outcome_name)

    # First, check if there is a directed path from action to outcome
    if not graph.has_directed_path(treatment_name, outcome_name):
        logger.warn("No directed path from treatment to outcome. Causal Effect is zero.")
        return IdentifiedEstimand(
            None,
            treatment_variable=treatment_name,
            outcome_variable=outcome_name,
            no_directed_path=True,
        )
    if estimand_type == EstimandType.NONPARAMETRIC_ATE:
        return identify_ate_effect(
            graph,
            treatment_name,
            outcome_name,
            backdoor_adjustment,
            optimize_backdoor,
            estimand_type,
            costs,
            conditional_node_names,
            proceed_when_unidentifiable,
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_NDE:
        return identify_nde_effect(
            graph, treatment_name, outcome_name, backdoor_adjustment, estimand_type, proceed_when_unidentifiable
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_NIE:
        return identify_nie_effect(
            graph, treatment_name, outcome_name, backdoor_adjustment, estimand_type, proceed_when_unidentifiable
        )
    elif estimand_type == EstimandType.NONPARAMETRIC_CDE:
        return identify_cde_effect(
            graph, treatment_name, outcome_name, backdoor_adjustment, estimand_type, proceed_when_unidentifiable
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
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: str,
    backdoor_adjustment: BackdoorAdjustment,
    optimize_backdoor: bool,
    estimand_type: EstimandType,
    costs: List,
    conditional_node_names: List[str] = None,
    proceed_when_unidentifiable: bool = False,
):
    estimands_dict = {}
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    ### 1. BACKDOOR IDENTIFICATION
    # Pick algorithm to compute backdoor sets according to method chosen
    if backdoor_adjustment not in EFFICIENT_METHODS:
        # First, checking if there are any valid backdoor adjustment sets
        if optimize_backdoor == False:
            backdoor_sets = identify_backdoor(graph, treatment_name, outcome_name, backdoor_adjustment)
        else:
            from dowhy.causal_identifier.backdoor import Backdoor

            path = Backdoor(graph._graph, treatment_name, outcome_name)
            backdoor_sets = path.get_backdoor_vars()
    elif backdoor_adjustment in EFFICIENT_METHODS:
        backdoor_sets = identify_efficient_backdoor(
            graph, backdoor_adjustment, costs, conditional_node_names=conditional_node_names
        )
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph, treatment_name, outcome_name, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    if len(backdoor_variables_dict) > 0:
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    else:
        estimands_dict["backdoor"] = None
    ### 2. INSTRUMENTAL VARIABLE IDENTIFICATION
    # Now checking if there is also a valid iv estimand
    instrument_names = graph.get_instruments(treatment_name, outcome_name)
    logger.info("Instrumental variables for treatment and outcome:" + str(instrument_names))
    if len(instrument_names) > 0:
        iv_estimand_expr = construct_iv_estimand(
            treatment_name,
            outcome_name,
            instrument_names,
        )
        logger.debug("Identified expression = " + str(iv_estimand_expr))
        estimands_dict["iv"] = iv_estimand_expr
    else:
        estimands_dict["iv"] = None

    ### 3. FRONTDOOR IDENTIFICATION
    # Now checking if there is a valid frontdoor variable
    frontdoor_variables_names = identify_frontdoor(graph, treatment_name, outcome_name)
    logger.info("Frontdoor variables for treatment and outcome:" + str(frontdoor_variables_names))
    if len(frontdoor_variables_names) > 0:
        frontdoor_estimand_expr = construct_frontdoor_estimand(
            treatment_name,
            outcome_name,
            frontdoor_variables_names,
        )
        logger.debug("Identified expression = " + str(frontdoor_estimand_expr))
        estimands_dict["frontdoor"] = frontdoor_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, treatment_name, outcome_name, frontdoor_variables_names, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, treatment_name, frontdoor_variables_names, outcome_name, backdoor_adjustment
        )
    else:
        estimands_dict["frontdoor"] = None

    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=treatment_name,
        outcome_variable=outcome_name,
        estimand_type=estimand_type,
        estimands=estimands_dict,
        backdoor_variables=backdoor_variables_dict,
        instrumental_variables=instrument_names,
        frontdoor_variables=frontdoor_variables_names,
        mediation_first_stage_confounders=mediation_first_stage_confounders,
        mediation_second_stage_confounders=mediation_second_stage_confounders,
        default_backdoor_id=default_backdoor_id,
    )
    return estimand


def identify_cde_effect(
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: str,
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
    proceed_when_unidentifiable: bool = False,
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
    backdoor_sets = identify_backdoor(graph, treatment_name, outcome_name, backdoor_adjustment, direct_effect=True)
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph, treatment_name, outcome_name, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    if len(backdoor_variables_dict) > 0:
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    else:
        estimands_dict["backdoor"] = None

    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=treatment_name,
        outcome_variable=outcome_name,
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
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: str,
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
    proceed_when_unidentifiable: bool = False,
):
    estimands_dict = {}
    ### 1. FIRST DOING BACKDOOR IDENTIFICATION
    # First, checking if there are any valid backdoor adjustment sets
    backdoor_sets = identify_backdoor(graph, treatment_name, outcome_name, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph, treatment_name, outcome_name, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

    ### 2. SECOND, CHECKING FOR MEDIATORS
    # Now checking if there are valid mediator variables
    estimands_dict = {}  # Need to reinitialize this dictionary to avoid including the backdoor sets
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    mediators_names = identify_mediation(graph, treatment_name, outcome_name)
    logger.info("Mediators for treatment and outcome:" + str(mediators_names))
    if len(mediators_names) > 0:
        mediation_estimand_expr = construct_mediation_estimand(
            estimand_type,
            treatment_name,
            outcome_name,
            mediators_names,
        )
        logger.debug("Identified expression = " + str(mediation_estimand_expr))
        estimands_dict["mediation"] = mediation_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, treatment_name, outcome_name, mediators_names, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, treatment_name, mediators_names, outcome_name, backdoor_adjustment
        )
    else:
        estimands_dict["mediation"] = None
    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=treatment_name,
        outcome_variable=outcome_name,
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
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: str,
    backdoor_adjustment: BackdoorAdjustment,
    estimand_type: EstimandType,
    proceed_when_unidentifiable: bool = False,
):
    estimands_dict = {}
    ### 1. FIRST DOING BACKDOOR IDENTIFICATION
    # First, checking if there are any valid backdoor adjustment sets
    backdoor_sets = identify_backdoor(graph, treatment_name, outcome_name, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph, treatment_name, outcome_name, backdoor_sets, estimands_dict
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

    ### 2. SECOND, CHECKING FOR MEDIATORS
    # Now checking if there are valid mediator variables
    estimands_dict = {}
    mediation_first_stage_confounders = None
    mediation_second_stage_confounders = None
    mediators_names = identify_mediation(graph, treatment_name, outcome_name)
    logger.info("Mediators for treatment and outcome:" + str(mediators_names))
    if len(mediators_names) > 0:
        mediation_estimand_expr = construct_mediation_estimand(
            estimand_type,
            treatment_name,
            outcome_name,
            mediators_names,
        )
        logger.debug("Identified expression = " + str(mediation_estimand_expr))
        estimands_dict["mediation"] = mediation_estimand_expr
        mediation_first_stage_confounders = identify_mediation_first_stage_confounders(
            graph, treatment_name, outcome_name, mediators_names, backdoor_adjustment
        )
        mediation_second_stage_confounders = identify_mediation_second_stage_confounders(
            graph, treatment_name, mediators_names, outcome_name, backdoor_adjustment
        )
    else:
        estimands_dict["mediation"] = None
    # Finally returning the estimand object
    estimand = IdentifiedEstimand(
        None,
        treatment_variable=treatment_name,
        outcome_variable=outcome_name,
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
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: str,
    backdoor_adjustment: BackdoorAdjustment,
    include_unobserved: bool = False,
    dseparation_algo: str = "default",
    direct_effect: bool = False,
):
    backdoor_sets = []
    backdoor_paths = None
    bdoor_graph = None
    if dseparation_algo == "naive":
        backdoor_paths = graph.get_backdoor_paths(treatment_name, outcome_name)
    elif dseparation_algo == "default":
        bdoor_graph = graph.do_surgery(
            treatment_name,
            target_node_names=outcome_name,
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
    check = graph.check_valid_backdoor_set(
        treatment_name,
        outcome_name,
        empty_set,
        backdoor_paths=backdoor_paths,
        new_graph=bdoor_graph,
        dseparation_algo=dseparation_algo,
    )
    if check["is_dseparated"]:
        backdoor_sets.append({"backdoor_set": empty_set})
        # If the method is `minimal-adjustment`, return the empty set right away.
        if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN:
            return backdoor_sets

    # Second, checking for all other sets of variables. If include_unobserved is false, then only observed variables are eligible.
    eligible_variables = (
        graph.get_all_nodes(include_unobserved=include_unobserved) - set(treatment_name) - set(outcome_name)
    )
    if direct_effect:
        # only remove descendants of Y
        # also allow any causes of Y that are not caused by T (for lower variance)
        eligible_variables -= graph.get_descendants(outcome_name)
    else:
        # remove descendants of T (mediators) and descendants of Y
        eligible_variables -= graph.get_descendants(treatment_name)
    # If var is d-separated from both treatment or outcome, it cannot
    # be a part of the backdoor set
    filt_eligible_variables = set()
    for var in eligible_variables:
        dsep_treat_var = graph.check_dseparation(treatment_name, parse_state(var), set())
        dsep_outcome_var = graph.check_dseparation(outcome_name, parse_state(var), set())
        if not dsep_outcome_var or not dsep_treat_var:
            filt_eligible_variables.add(var)
    if backdoor_adjustment in METHOD_NAMES:
        backdoor_sets, found_valid_adjustment_set = find_valid_adjustment_sets(
            graph,
            treatment_name,
            outcome_name,
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
                treatment_name,
                outcome_name,
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
    graph: CausalGraph,
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
        conditional_node_names=conditional_node_names,
        costs=costs,
    )
    if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_EFFICIENT:
        backdoor_set = efficient_bd.optimal_adj_set()
        backdoor_sets = [{"backdoor_set": tuple(backdoor_set)}]
    elif backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN_EFFICIENT:
        backdoor_set = efficient_bd.optimal_minimal_adj_set()
        backdoor_sets = [{"backdoor_set": tuple(backdoor_set)}]
    elif backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT:
        backdoor_set = efficient_bd.optimal_mincost_adj_set()
        backdoor_sets = [{"backdoor_set": tuple(backdoor_set)}]
    return backdoor_sets


def find_valid_adjustment_sets(
    graph: CausalGraph,
    treatment_name: List,
    outcome_name: List,
    backdoor_paths: List,
    bdoor_graph: CausalGraph,
    dseparation_algo: str,
    backdoor_sets: List,
    filt_eligible_variables: List,
    backdoor_adjustment: BackdoorAdjustment,
    max_iterations: int,
):
    num_iterations = 0
    found_valid_adjustment_set = False
    all_nodes_observed = graph.all_observed(graph.get_all_nodes())
    # If `minimal-adjustment` method is specified, start the search from the set with minimum size. Otherwise, start from the largest.
    set_sizes = (
        range(1, len(filt_eligible_variables) + 1, 1)
        if backdoor_adjustment == BackdoorAdjustment.BACKDOOR_MIN
        else range(len(filt_eligible_variables), 0, -1)
    )
    for size_candidate_set in set_sizes:
        for candidate_set in itertools.combinations(filt_eligible_variables, size_candidate_set):
            check = graph.check_valid_backdoor_set(
                treatment_name,
                outcome_name,
                candidate_set,
                backdoor_paths=backdoor_paths,
                new_graph=bdoor_graph,
                dseparation_algo=dseparation_algo,
            )
            logger.debug(
                "Candidate backdoor set: {0}, is_dseparated: {1}".format(candidate_set, check["is_dseparated"])
            )
            if check["is_dseparated"]:
                backdoor_sets.append({"backdoor_set": candidate_set})
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
            and all_nodes_observed
        ):
            break
        if num_iterations > max_iterations:
            logger.warning(f"Max number of iterations {max_iterations} reached. Could not find a valid backdoor set.")
            break
    return backdoor_sets, found_valid_adjustment_set


def get_default_backdoor_set_id(
    graph: CausalGraph, treatment_name: List[str], outcome_name: List[str], backdoor_sets_dict: Dict
):
    # Adding a None estimand if no backdoor set found
    if len(backdoor_sets_dict) == 0:
        return None

    # Default set contains minimum possible number of instrumental variables, to prevent lowering variance in the treatment variable.
    instrument_names = set(graph.get_instruments(treatment_name, outcome_name))
    iv_count_dict = {
        key: len(set(bdoor_set).intersection(instrument_names)) for key, bdoor_set in backdoor_sets_dict.items()
    }
    min_iv_count = min(iv_count_dict.values())
    min_iv_keys = {key for key, iv_count in iv_count_dict.items() if iv_count == min_iv_count}
    min_iv_backdoor_sets_dict = {key: backdoor_sets_dict[key] for key in min_iv_keys}

    # Default set is the one with the least number of adjustment variables (optimizing for efficiency)
    min_set_length = 1000000
    default_key = None
    for key, bdoor_set in min_iv_backdoor_sets_dict.items():
        if len(bdoor_set) < min_set_length:
            min_set_length = len(bdoor_set)
            default_key = key
    return default_key


def build_backdoor_estimands_dict(
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: List[str],
    backdoor_sets: List[str],
    estimands_dict: Dict,
):
    """Build the final dict for backdoor sets by filtering unobserved variables if needed."""
    backdoor_variables_dict = {}
    is_identified = [graph.all_observed(bset["backdoor_set"]) for bset in backdoor_sets]

    if any(is_identified):
        logger.info("Causal effect can be identified.")
        backdoor_sets_arr = [
            list(bset["backdoor_set"]) for bset in backdoor_sets if graph.all_observed(bset["backdoor_set"])
        ]
    else:  # there is unobserved confounding
        logger.warning("Backdoor identification failed.")
        backdoor_sets_arr = []

    for i in range(len(backdoor_sets_arr)):
        backdoor_estimand_expr = construct_backdoor_estimand(treatment_name, outcome_name, backdoor_sets_arr[i])
        logger.debug("Identified expression = " + str(backdoor_estimand_expr))
        estimands_dict["backdoor" + str(i + 1)] = backdoor_estimand_expr
        backdoor_variables_dict["backdoor" + str(i + 1)] = backdoor_sets_arr[i]
    return estimands_dict, backdoor_variables_dict


def identify_frontdoor(
    graph: CausalGraph, treatment_name: List[str], outcome_name: List[str], dseparation_algo: str = "default"
):
    """Find a valid frontdoor variable if it exists.

    Currently only supports a single variable frontdoor set.
    """
    frontdoor_var = None
    frontdoor_paths = None
    fdoor_graph = None
    if dseparation_algo == "default":
        cond1_graph = graph.do_surgery(treatment_name, remove_incoming_edges=True)
        bdoor_graph1 = graph.do_surgery(treatment_name, remove_outgoing_edges=True)
    elif dseparation_algo == "naive":
        frontdoor_paths = graph.get_all_directed_paths(treatment_name, outcome_name)
    else:
        raise ValueError(f"d-separation algorithm {dseparation_algo} is not supported")

    eligible_variables = (
        graph.get_descendants(treatment_name) - set(outcome_name) - set(graph.get_descendants(outcome_name))
    )
    # For simplicity, assuming a one-variable frontdoor set
    for candidate_var in eligible_variables:
        # Cond 1: All directed paths intercepted by candidate_var
        cond1 = graph.check_valid_frontdoor_set(
            treatment_name,
            outcome_name,
            parse_state(candidate_var),
            frontdoor_paths=frontdoor_paths,
            new_graph=cond1_graph,
            dseparation_algo=dseparation_algo,
        )
        logger.debug("Candidate frontdoor set: {0}, is_dseparated: {1}".format(candidate_var, cond1))
        if not cond1:
            continue
        # Cond 2: No confounding between treatment and candidate var
        cond2 = graph.check_valid_backdoor_set(
            treatment_name,
            parse_state(candidate_var),
            set(),
            backdoor_paths=None,
            new_graph=bdoor_graph1,
            dseparation_algo=dseparation_algo,
        )
        if not cond2:
            continue
        # Cond 3: treatment blocks all confounding between candidate_var and outcome
        bdoor_graph2 = graph.do_surgery(candidate_var, remove_outgoing_edges=True)
        cond3 = graph.check_valid_backdoor_set(
            parse_state(candidate_var),
            outcome_name,
            treatment_name,
            backdoor_paths=None,
            new_graph=bdoor_graph2,
            dseparation_algo=dseparation_algo,
        )
        is_valid_frontdoor = cond1 and cond2 and cond3
        if is_valid_frontdoor:
            frontdoor_var = candidate_var
            break
    return parse_state(frontdoor_var)


def identify_mediation(graph: CausalGraph, treatment_name: List[str], outcome_name: List[str]):
    """Find a valid mediator if it exists.

    Currently only supports a single variable mediator set.
    """
    mediation_var = None
    mediation_paths = graph.get_all_directed_paths(treatment_name, outcome_name)
    eligible_variables = graph.get_descendants(treatment_name) - set(outcome_name)
    # For simplicity, assuming a one-variable mediation set
    for candidate_var in eligible_variables:
        is_valid_mediation = graph.check_valid_mediation_set(
            treatment_name,
            outcome_name,
            parse_state(candidate_var),
            mediation_paths=mediation_paths,
        )
        logger.debug("Candidate mediation set: {0}, on_mediating_path: {1}".format(candidate_var, is_valid_mediation))
        if is_valid_mediation:
            mediation_var = candidate_var
            break
    return parse_state(mediation_var)


def identify_mediation_first_stage_confounders(
    graph: CausalGraph,
    treatment_name: List[str],
    outcome_name: List[str],
    mediators_names: List[str],
    backdoor_adjustment: BackdoorAdjustment,
):
    # Create estimands dict as per the API for backdoor, but do not return it
    estimands_dict = {}
    backdoor_sets = identify_backdoor(graph, treatment_name, mediators_names, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph,
        treatment_name,
        mediators_names,
        backdoor_sets,
        estimands_dict,
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    return backdoor_variables_dict


def identify_mediation_second_stage_confounders(
    graph: CausalGraph,
    treatment_name: List[str],
    mediators_names: List[str],
    outcome_name: List[str],
    backdoor_adjustment: BackdoorAdjustment,
):
    # Create estimands dict as per the API for backdoor, but do not return it
    estimands_dict = {}
    backdoor_sets = identify_backdoor(graph, mediators_names, outcome_name, backdoor_adjustment)
    estimands_dict, backdoor_variables_dict = build_backdoor_estimands_dict(
        graph,
        mediators_names,
        outcome_name,
        backdoor_sets,
        estimands_dict,
    )
    # Setting default "backdoor" identification adjustment set
    default_backdoor_id = get_default_backdoor_set_id(graph, treatment_name, outcome_name, backdoor_variables_dict)
    estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
    backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
    return backdoor_variables_dict


def construct_backdoor_estimand(treatment_name: List[str], outcome_name: List[str], common_causes: List[str]):
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
    estimand_type: EstimandType, treatment_name: List[str], outcome_name: List[str], mediators_names: List[str]
):
    # TODO: support multivariate treatments better.
    expr = None
    if estimand_type in (
        EstimandType.NONPARAMETRIC_NDE,
        EstimandType.NONPARAMETRIC_NIE,
    ):
        outcome_name = outcome_name[0]
        sym_outcome = spstats.Normal(outcome_name, 0, 1)
        sym_treatment_symbols = [spstats.Normal(t, 0, 1) for t in treatment_name]
        sym_treatment = sp.Array(sym_treatment_symbols)
        sym_mediators_symbols = [sp.Symbol(inst) for inst in mediators_names]
        sym_mediators = sp.Array(sym_mediators_symbols)
        sym_outcome_derivative = sp.Derivative(sym_outcome, sym_mediators)
        sym_treatment_derivative = sp.Derivative(sym_mediators, sym_treatment)
        # For direct effect
        num_expr_str = outcome_name
        if len(mediators_names) > 0:
            num_expr_str += "|" + ",".join(mediators_names)
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
                ",".join(treatment_name),
                ",".join(outcome_name),
                ",".join(mediators_names),
            ),
            "First-stage-unconfoundedness": (
                "If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}" " then P({1}|{0},U) = P({1}|{0})"
            ).format(",".join(treatment_name), ",".join(mediators_names)),
            "Second-stage-unconfoundedness": (
                "If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}"
                " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
            ).format(",".join(treatment_name), outcome_name, ",".join(mediators_names)),
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
