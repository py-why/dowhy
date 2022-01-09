import copy
import itertools
import logging

import sympy as sp
import sympy.stats as spstats

import dowhy.utils.cli_helpers as cli
from dowhy.utils.api import parse_state


class CausalIdentifier:

    """Class that implements different identification methods.

    Currently supports backdoor and instrumental variable identification methods. The identification is based on the causal graph provided.

    Other specific ways of identification, such as the ID* algorithm, minimal adjustment criteria, etc. will be added in the future.
    If you'd like to contribute, please raise an issue or a pull request on Github.

    """
    NONPARAMETRIC_ATE="nonparametric-ate"
    NONPARAMETRIC_NDE="nonparametric-nde"
    NONPARAMETRIC_NIE="nonparametric-nie"
    MAX_BACKDOOR_ITERATIONS = 100000

    # Backdoor method names
    BACKDOOR_DEFAULT="default"
    BACKDOOR_EXHAUSTIVE="exhaustive-search"
    BACKDOOR_MIN="minimal-adjustment"
    BACKDOOR_MAX="maximal-adjustment"
    METHOD_NAMES = {BACKDOOR_DEFAULT, BACKDOOR_EXHAUSTIVE, BACKDOOR_MIN, BACKDOOR_MAX}
    DEFAULT_BACKDOOR_METHOD = BACKDOOR_DEFAULT

    def __init__(self, graph, estimand_type,
            method_name = "default",
            proceed_when_unidentifiable=False):
        self._graph = graph
        self.estimand_type = estimand_type
        self.treatment_name = graph.treatment_name
        self.outcome_name = graph.outcome_name
        self.method_name = method_name
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self.logger = logging.getLogger(__name__)

    def identify_effect(self, optimize_backdoor=False):
        """Main method that returns an identified estimand (if one exists).

        If estimand_type is non-parametric ATE, then  uses backdoor, instrumental variable and frontdoor identification methods,  to check if an identified estimand exists, based on the causal graph.

        :param self: instance of the CausalIdentifier class (or its subclass)
        :returns:  target estimand, an instance of the IdentifiedEstimand class
        """
        # First, check if there is a directed path from action to outcome
        if not self._graph.has_directed_path(self.treatment_name, self.outcome_name):
            self.logger.warn("No directed path from treatment to outcome. Causal Effect is zero.")
            return IdentifiedEstimand(self,
                    treatment_variable=self.treatment_name,
                    outcome_variable=self.outcome_name,
                    no_directed_path=True)
        if self.estimand_type == CausalIdentifier.NONPARAMETRIC_ATE:
            return self.identify_ate_effect(optimize_backdoor=optimize_backdoor)
        elif self.estimand_type == CausalIdentifier.NONPARAMETRIC_NDE:
            return self.identify_nde_effect()
        elif self.estimand_type == CausalIdentifier.NONPARAMETRIC_NIE:
            return self.identify_nie_effect()
        else:
            raise ValueError("Estimand type is not supported. Use either {0}, {1}, or {2}.".format(
                CausalIdentifier.NONPARAMETRIC_ATE,
                CausalIdentifier.NONPARAMETRIC_NDE,
                CausalIdentifier.NONPARAMETRIC_NIE))

    def identify_ate_effect(self, optimize_backdoor):
        estimands_dict = {}
        mediation_first_stage_confounders = None
        mediation_second_stage_confounders = None
        ### 1. BACKDOOR IDENTIFICATION
        # First, checking if there are any valid backdoor adjustment sets
        if optimize_backdoor == False:
            backdoor_sets = self.identify_backdoor(self.treatment_name, self.outcome_name)
        else:
            from dowhy.causal_identifiers.backdoor import Backdoor
            path = Backdoor(self._graph._graph, self.treatment_name, self.outcome_name)
            backdoor_sets = path.get_backdoor_vars()
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                self.treatment_name,
                self.outcome_name,
                backdoor_sets,
                estimands_dict)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        if len(backdoor_variables_dict) > 0:
            estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
            backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
        else:
            estimands_dict["backdoor"] = None
        ### 2. INSTRUMENTAL VARIABLE IDENTIFICATION
        # Now checking if there is also a valid iv estimand
        instrument_names = self._graph.get_instruments(self.treatment_name,
                                                       self.outcome_name)
        self.logger.info("Instrumental variables for treatment and outcome:" +
                         str(instrument_names))
        if len(instrument_names) > 0:
            iv_estimand_expr = self.construct_iv_estimand(
                self.estimand_type,
                self._graph.treatment_name,
                self._graph.outcome_name,
                instrument_names
            )
            self.logger.debug("Identified expression = " + str(iv_estimand_expr))
            estimands_dict["iv"] = iv_estimand_expr
        else:
            estimands_dict["iv"] = None

        ### 3. FRONTDOOR IDENTIFICATION
        # Now checking if there is a valid frontdoor variable
        frontdoor_variables_names = self.identify_frontdoor()
        self.logger.info("Frontdoor variables for treatment and outcome:" +
                str(frontdoor_variables_names))
        if len(frontdoor_variables_names) >0:
            frontdoor_estimand_expr = self.construct_frontdoor_estimand(
                self.estimand_type,
                self._graph.treatment_name,
                self._graph.outcome_name,
                frontdoor_variables_names
            )
            self.logger.debug("Identified expression = " + str(frontdoor_estimand_expr))
            estimands_dict["frontdoor"] = frontdoor_estimand_expr
            mediation_first_stage_confounders = self.identify_mediation_first_stage_confounders(self.treatment_name, frontdoor_variables_names)
            mediation_second_stage_confounders = self.identify_mediation_second_stage_confounders(frontdoor_variables_names, self.outcome_name)
        else:
            estimands_dict["frontdoor"] = None

        # Finally returning the estimand object
        estimand = IdentifiedEstimand(
            self,
            treatment_variable=self._graph.treatment_name,
            outcome_variable=self._graph.outcome_name,
            estimand_type=self.estimand_type,
            estimands=estimands_dict,
            backdoor_variables=backdoor_variables_dict,
            instrumental_variables=instrument_names,
            frontdoor_variables=frontdoor_variables_names,
            mediation_first_stage_confounders=mediation_first_stage_confounders,
            mediation_second_stage_confounders=mediation_second_stage_confounders,
            default_backdoor_id = default_backdoor_id
        )
        return estimand

    def identify_nie_effect(self):
        estimands_dict = {}
        ### 1. FIRST DOING BACKDOOR IDENTIFICATION
        # First, checking if there are any valid backdoor adjustment sets
        backdoor_sets = self.identify_backdoor(self.treatment_name, self.outcome_name)
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                self.treatment_name,
                self.outcome_name,
                backdoor_sets,
                estimands_dict)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

        ### 2. SECOND, CHECKING FOR MEDIATORS
        # Now checking if there are valid mediator variables
        estimands_dict = {} # Need to reinitialize this dictionary to avoid including the backdoor sets
        mediation_first_stage_confounders = None
        mediation_second_stage_confounders = None
        mediators_names = self.identify_mediation()
        self.logger.info("Mediators for treatment and outcome:" +
                str(mediators_names))
        if len(mediators_names) >0:
            mediation_estimand_expr = self.construct_mediation_estimand(
                self.estimand_type,
                self._graph.treatment_name,
                self._graph.outcome_name,
                mediators_names
            )
            self.logger.debug("Identified expression = " + str(mediation_estimand_expr))
            estimands_dict["mediation"] = mediation_estimand_expr
            mediation_first_stage_confounders = self.identify_mediation_first_stage_confounders(self.treatment_name, mediators_names)
            mediation_second_stage_confounders = self.identify_mediation_second_stage_confounders(mediators_names, self.outcome_name)
        else:
            estimands_dict["mediation"] = None
        # Finally returning the estimand object
        estimand = IdentifiedEstimand(
            self,
            treatment_variable=self._graph.treatment_name,
            outcome_variable=self._graph.outcome_name,
            estimand_type=self.estimand_type,
            estimands=estimands_dict,
            backdoor_variables=backdoor_variables_dict,
            instrumental_variables=None,
            frontdoor_variables=None,
            mediator_variables=mediators_names,
            mediation_first_stage_confounders=mediation_first_stage_confounders,
            mediation_second_stage_confounders=mediation_second_stage_confounders,
            default_backdoor_id = None
        )
        return estimand

    def identify_nde_effect(self):
        estimands_dict = {}
        ### 1. FIRST DOING BACKDOOR IDENTIFICATION
        # First, checking if there are any valid backdoor adjustment sets
        backdoor_sets = self.identify_backdoor(self.treatment_name, self.outcome_name)
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                self.treatment_name,
                self.outcome_name,
                backdoor_sets,
                estimands_dict)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

        ### 2. SECOND, CHECKING FOR MEDIATORS
        # Now checking if there are valid mediator variables
        estimands_dict = {}
        mediation_first_stage_confounders = None
        mediation_second_stage_confounders = None
        mediators_names = self.identify_mediation()
        self.logger.info("Mediators for treatment and outcome:" +
                str(mediators_names))
        if len(mediators_names) >0:
            mediation_estimand_expr = self.construct_mediation_estimand(
                self.estimand_type,
                self._graph.treatment_name,
                self._graph.outcome_name,
                mediators_names
            )
            self.logger.debug("Identified expression = " + str(mediation_estimand_expr))
            estimands_dict["mediation"] = mediation_estimand_expr
            mediation_first_stage_confounders = self.identify_mediation_first_stage_confounders(self.treatment_name, mediators_names)
            mediation_second_stage_confounders = self.identify_mediation_second_stage_confounders(mediators_names, self.outcome_name)
        else:
            estimands_dict["mediation"] = None
        # Finally returning the estimand object
        estimand = IdentifiedEstimand(
            self,
            treatment_variable=self._graph.treatment_name,
            outcome_variable=self._graph.outcome_name,
            estimand_type=self.estimand_type,
            estimands=estimands_dict,
            backdoor_variables=backdoor_variables_dict,
            instrumental_variables=None,
            frontdoor_variables=None,
            mediator_variables=mediators_names,
            mediation_first_stage_confounders=mediation_first_stage_confounders,
            mediation_second_stage_confounders=mediation_second_stage_confounders,
            default_backdoor_id = None
        )
        return estimand

    def identify_backdoor(self, treatment_name, outcome_name,
            include_unobserved=False, dseparation_algo="default"):
        backdoor_sets = []
        backdoor_paths = None
        bdoor_graph = None
        if dseparation_algo == "naive":
            backdoor_paths = self._graph.get_backdoor_paths(treatment_name, outcome_name)
        elif dseparation_algo == "default":
            bdoor_graph = self._graph.do_surgery(treatment_name,
                    remove_outgoing_edges=True)
        else:
            raise ValueError(f"d-separation algorithm {dseparation_algo} is not supported")
        method_name = self.method_name if self.method_name != CausalIdentifier.BACKDOOR_DEFAULT else CausalIdentifier.DEFAULT_BACKDOOR_METHOD

        # First, checking if empty set is a valid backdoor set
        empty_set = set()
        check = self._graph.check_valid_backdoor_set(treatment_name,
                outcome_name, empty_set,
                backdoor_paths=backdoor_paths, new_graph=bdoor_graph,
                dseparation_algo=dseparation_algo)
        if check["is_dseparated"]:
            backdoor_sets.append({'backdoor_set':empty_set})
            # If the method is `minimal-adjustment`, return the empty set right away.
            if method_name == CausalIdentifier.BACKDOOR_MIN:
                return backdoor_sets

        # Second, checking for all other sets of variables. If include_unobserved is false, then only observed variables are eligible.
        eligible_variables = self._graph.get_all_nodes(include_unobserved=include_unobserved) \
            - set(treatment_name) \
            - set(outcome_name)
        eligible_variables -= self._graph.get_descendants(treatment_name)
        # If var is d-separated from both treatment or outcome, it cannot
        # be a part of the backdoor set
        filt_eligible_variables = set()
        for var in eligible_variables:
            dsep_treat_var = self._graph.check_dseparation(
                    treatment_name, parse_state(var),
                    set())
            dsep_outcome_var = self._graph.check_dseparation(
                    outcome_name, parse_state(var), set())
            if not dsep_outcome_var or not dsep_treat_var:
                filt_eligible_variables.add(var)
        if method_name in CausalIdentifier.METHOD_NAMES:
            backdoor_sets, found_valid_adjustment_set = self.find_valid_adjustment_sets(
                    treatment_name, outcome_name,
                    backdoor_paths, bdoor_graph,
                    dseparation_algo,
                    backdoor_sets, filt_eligible_variables,
                    method_name=method_name,
                    max_iterations= CausalIdentifier.MAX_BACKDOOR_ITERATIONS)
            if method_name == CausalIdentifier.BACKDOOR_DEFAULT and found_valid_adjustment_set:
                # repeat the above search with BACKDOOR_MIN
                backdoor_sets, _ = self.find_valid_adjustment_sets(
                        treatment_name, outcome_name,
                        backdoor_paths, bdoor_graph,
                        dseparation_algo,
                        backdoor_sets, filt_eligible_variables,
                        method_name=CausalIdentifier.BACKDOOR_MIN,
                        max_iterations= CausalIdentifier.MAX_BACKDOOR_ITERATIONS)
        else:
            raise ValueError(f"Identifier method {method_name} not supported. Try one of the following: {CausalIdentifier.METHOD_NAMES}")
        return backdoor_sets

    def find_valid_adjustment_sets(self, treatment_name, outcome_name,
            backdoor_paths, bdoor_graph, dseparation_algo,
            backdoor_sets, filt_eligible_variables,
            method_name, max_iterations):
        num_iterations = 0
        found_valid_adjustment_set = False
        all_nodes_observed = self._graph.all_observed(self._graph.get_all_nodes())
        # If `minimal-adjustment` method is specified, start the search from the set with minimum size. Otherwise, start from the largest.
        set_sizes = range(1, len(filt_eligible_variables) + 1, 1) if method_name == CausalIdentifier.BACKDOOR_MIN else range(len(filt_eligible_variables), 0, -1)
        for size_candidate_set in set_sizes:
            for candidate_set in itertools.combinations(filt_eligible_variables, size_candidate_set):
                check = self._graph.check_valid_backdoor_set(treatment_name,
                        outcome_name, candidate_set,
                        backdoor_paths=backdoor_paths,
                        new_graph = bdoor_graph,
                        dseparation_algo = dseparation_algo)
                self.logger.debug("Candidate backdoor set: {0}, is_dseparated: {1}".format(candidate_set, check["is_dseparated"]))
                if check["is_dseparated"]:
                    backdoor_sets.append({'backdoor_set': candidate_set})
                    found_valid_adjustment_set = True
                num_iterations += 1
                if method_name == CausalIdentifier.BACKDOOR_EXHAUSTIVE and num_iterations > max_iterations:
                    self.logger.warning(f"Max number of iterations {max_iterations} reached.")
                    break
            # If the backdoor method is `maximal-adjustment` or `minimal-adjustment`, return the first found adjustment set.
            if method_name in {CausalIdentifier.BACKDOOR_DEFAULT, CausalIdentifier.BACKDOOR_MAX, CausalIdentifier.BACKDOOR_MIN} and found_valid_adjustment_set:
                break
            # If all variables are observed, and the biggest eligible set
            # does not satisfy backdoor, then none of its subsets will.
            if method_name in {CausalIdentifier.BACKDOOR_DEFAULT, CausalIdentifier.BACKDOOR_MAX} and all_nodes_observed:
                break
            if num_iterations > max_iterations:
                self.logger.warning(f"Max number of iterations {max_iterations} reached. Could not find a valid backdoor set.")
                break
        return backdoor_sets, found_valid_adjustment_set


    def get_default_backdoor_set_id(self, backdoor_sets_dict):
        # Adding a None estimand if no backdoor set found
        if len(backdoor_sets_dict) == 0:
            return None

        # Default set contains minimum possible number of instrumental variables, to prevent lowering variance in the treatment variable.
        instrument_names = set(self._graph.get_instruments(self.treatment_name, self.outcome_name))
        iv_count_dict = {key: len(set(bdoor_set).intersection(instrument_names)) for key, bdoor_set in backdoor_sets_dict.items()}
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

    def build_backdoor_estimands_dict(self, treatment_name, outcome_name,
            backdoor_sets, estimands_dict, proceed_when_unidentifiable=None):
        """Build the final dict for backdoor sets by filtering unobserved variables if needed.
        """
        backdoor_variables_dict = {}
        if proceed_when_unidentifiable is None:
            proceed_when_unidentifiable = self._proceed_when_unidentifiable
        is_identified = [ self._graph.all_observed(bset["backdoor_set"]) for bset in backdoor_sets ]

        if any(is_identified):
            self.logger.info("Causal effect can be identified.")
            backdoor_sets_arr = [list(
                bset["backdoor_set"])
                for bset in backdoor_sets
                if self._graph.all_observed(bset["backdoor_set"]) ]
        else: # there is unobserved confounding
            self.logger.warning("Backdoor identification failed.")
            backdoor_sets_arr = []

        for i in range(len(backdoor_sets_arr)):
            backdoor_estimand_expr = self.construct_backdoor_estimand(
                self.estimand_type, treatment_name,
                outcome_name, backdoor_sets_arr[i])
            self.logger.debug("Identified expression = " + str(backdoor_estimand_expr))
            estimands_dict["backdoor"+str(i+1)] = backdoor_estimand_expr
            backdoor_variables_dict["backdoor"+str(i+1)] = backdoor_sets_arr[i]
        return estimands_dict, backdoor_variables_dict

    def identify_frontdoor(self, dseparation_algo="default"):
        """ Find a valid frontdoor variable if it exists.

        Currently only supports a single variable frontdoor set.
        """
        frontdoor_var = None
        frontdoor_paths = None
        fdoor_graph = None
        if dseparation_algo == "default":
            cond1_graph = self._graph.do_surgery(self.treatment_name,
                    remove_incoming_edges=True)
            bdoor_graph1 = self._graph.do_surgery(self.treatment_name,
                    remove_outgoing_edges=True)
        elif dseparation_algo == "naive":
            frontdoor_paths = self._graph.get_all_directed_paths(self.treatment_name, self.outcome_name)
        else:
            raise ValueError(f"d-separation algorithm {dseparation_algo} is not supported")


        eligible_variables = self._graph.get_descendants(self.treatment_name) \
            - set(self.outcome_name) \
            - set(self._graph.get_descendants(self.outcome_name))
        # For simplicity, assuming a one-variable frontdoor set
        for candidate_var in eligible_variables:
            # Cond 1: All directed paths intercepted by candidate_var
            cond1 = self._graph.check_valid_frontdoor_set(
                self.treatment_name, self.outcome_name,
                parse_state(candidate_var),
                frontdoor_paths=frontdoor_paths,
                new_graph=cond1_graph,
                dseparation_algo=dseparation_algo)
            self.logger.debug("Candidate frontdoor set: {0}, is_dseparated: {1}".format(candidate_var, cond1))
            if not cond1:
                continue
            # Cond 2: No confounding between treatment and candidate var
            cond2 = self._graph.check_valid_backdoor_set(
                self.treatment_name, parse_state(candidate_var),
                set(),
                backdoor_paths=None,
                new_graph= bdoor_graph1,
                dseparation_algo=dseparation_algo)
            if not cond2:
                continue
            # Cond 3: treatment blocks all confounding between candidate_var and outcome
            bdoor_graph2 = self._graph.do_surgery(candidate_var,
                    remove_outgoing_edges=True)
            cond3 = self._graph.check_valid_backdoor_set(
                parse_state(candidate_var), self.outcome_name,
                self.treatment_name,
                backdoor_paths=None,
                new_graph= bdoor_graph2,
                dseparation_algo=dseparation_algo)
            is_valid_frontdoor = cond1 and cond2 and cond3
            if is_valid_frontdoor:
                frontdoor_var = candidate_var
                break
        return parse_state(frontdoor_var)

    def identify_mediation(self):
        """ Find a valid mediator if it exists.

        Currently only supports a single variable mediator set.
        """
        mediation_var = None
        mediation_paths = self._graph.get_all_directed_paths(self.treatment_name, self.outcome_name)
        eligible_variables = self._graph.get_descendants(self.treatment_name) \
            - set(self.outcome_name)
        # For simplicity, assuming a one-variable mediation set
        for candidate_var in eligible_variables:
            is_valid_mediation = self._graph.check_valid_mediation_set(self.treatment_name,
                    self.outcome_name, parse_state(candidate_var), mediation_paths=mediation_paths)
            self.logger.debug("Candidate mediation set: {0}, on_mediating_path: {1}".format(candidate_var, is_valid_mediation))
            if is_valid_mediation:
                mediation_var = candidate_var
                break
        return parse_state(mediation_var)


        return None

    def identify_mediation_first_stage_confounders(self, treatment_name, mediators_names):
        # Create estimands dict as per the API for backdoor, but do not return it
        estimands_dict = {}
        backdoor_sets = self.identify_backdoor(treatment_name, mediators_names)
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                treatment_name,
                mediators_names,
                backdoor_sets,
                estimands_dict,
                proceed_when_unidentifiable=True)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
        return backdoor_variables_dict

    def identify_mediation_second_stage_confounders(self, mediators_names, outcome_name):
        # Create estimands dict as per the API for backdoor, but do not return it
        estimands_dict = {}
        backdoor_sets = self.identify_backdoor(mediators_names, outcome_name)
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                mediators_names,
                outcome_name,
                backdoor_sets,
                estimands_dict,
                proceed_when_unidentifiable=True)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)
        return backdoor_variables_dict

    def construct_backdoor_estimand(self, estimand_type, treatment_name,
                                    outcome_name, common_causes):
        # TODO: outputs string for now, but ideally should do symbolic
        # expressions Mon 19 Feb 2018 04:54:17 PM DST
        # TODO Better support for multivariate treatments

        expr = None
        outcome_name = outcome_name[0]
        num_expr_str = outcome_name
        if len(common_causes)>0:
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
            'Unconfoundedness': (
                u"If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{1}"
                " then P({1}|{0},{2},U) = P({1}|{0},{2})"
            ).format(",".join(treatment_name), outcome_name, ",".join(common_causes))
        }

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand

    def construct_iv_estimand(self, estimand_type, treatment_name,
                              outcome_name, instrument_names):
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
                u"If we remove {{{0}}}\N{RIGHTWARDS ARROW}{{{1}}}, then "
                u"\N{NOT SIGN}({{{0}}}\N{RIGHTWARDS ARROW}{2})"
            ).format(",".join(instrument_names), ",".join(treatment_name),
                     outcome_name)
        }

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand

    def construct_frontdoor_estimand(self, estimand_type, treatment_name,
                              outcome_name, frontdoor_variables_names):
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
            "Full-mediation": (
                "{2} intercepts (blocks) all directed paths from {0} to {1}."
            ).format(",".join(treatment_name), ",".join(outcome_name), ",".join(frontdoor_variables_names)),
            "First-stage-unconfoundedness": (
                u"If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}"
                " then P({1}|{0},U) = P({1}|{0})"
            ).format(",".join(treatment_name), ",".join(frontdoor_variables_names)),
            "Second-stage-unconfoundedness": (
                u"If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}"
                " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
            ).format(",".join(treatment_name), outcome_name, ",".join(frontdoor_variables_names))
        }

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand

    def construct_mediation_estimand(self, estimand_type, treatment_name,
                              outcome_name, mediators_names):
        # TODO: support multivariate treatments better.
        expr = None
        if estimand_type in (CausalIdentifier.NONPARAMETRIC_NDE, CausalIdentifier.NONPARAMETRIC_NIE):
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
            if len(mediators_names)>0:
                num_expr_str += "|" + ",".join(mediators_names)
            sym_mu = sp.Symbol("mu")
            sym_sigma = sp.Symbol("sigma", positive=True)
            sym_conditional_outcome = spstats.Normal(num_expr_str, sym_mu, sym_sigma)
            sym_directeffect_derivative = sp.Derivative(sym_conditional_outcome, sym_treatment)
            if estimand_type == CausalIdentifier.NONPARAMETRIC_NIE:
                sym_effect = spstats.Expectation(sym_treatment_derivative * sym_outcome_derivative)
            elif estimand_type == CausalIdentifier.NONPARAMETRIC_NDE:
                sym_effect = spstats.Expectation(sym_directeffect_derivative)
            sym_assumptions = {
                "Mediation": (
                    "{2} intercepts (blocks) all directed paths from {0} to {1} except the path {{{0}}}\N{RIGHTWARDS ARROW}{{{1}}}."
                ).format(",".join(treatment_name), ",".join(outcome_name), ",".join(mediators_names)),
                "First-stage-unconfoundedness": (
                    u"If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}"
                    " then P({1}|{0},U) = P({1}|{0})"
                ).format(",".join(treatment_name), ",".join(mediators_names)),
                "Second-stage-unconfoundedness": (
                    u"If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}"
                    " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
                ).format(",".join(treatment_name), outcome_name, ",".join(mediators_names))
            }
        else:
            raise ValueError("Estimand type not supported. Supported estimand types are {0} or {1}'.".format(
                CausalIdentifier.NONPARAMETRIC_NDE,
                CausalIdentifier.NONPARAMETRIC_NIE))

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand


class IdentifiedEstimand:

    """Class for storing a causal estimand, typically as a result of the identification step.

    """

    def __init__(self, identifier, treatment_variable, outcome_variable,
                 estimand_type=None, estimands=None,
                 backdoor_variables=None, instrumental_variables=None,
                 frontdoor_variables=None,
                 mediator_variables=None,
                 mediation_first_stage_confounders=None,
                 mediation_second_stage_confounders=None,
                 default_backdoor_id=None, identifier_method=None,
                 no_directed_path=False):
        self.identifier = identifier
        self.treatment_variable = parse_state(treatment_variable)
        self.outcome_variable = parse_state(outcome_variable)
        self.backdoor_variables = backdoor_variables
        self.instrumental_variables = parse_state(instrumental_variables)
        self.frontdoor_variables = parse_state(frontdoor_variables)
        self.mediator_variables = parse_state(mediator_variables)
        self.mediation_first_stage_confounders=mediation_first_stage_confounders
        self.mediation_second_stage_confounders=mediation_second_stage_confounders
        self.estimand_type = estimand_type
        self.estimands = estimands
        self.default_backdoor_id = default_backdoor_id
        self.identifier_method = identifier_method
        self.no_directed_path = no_directed_path

    def set_identifier_method(self, identifier_name):
        self.identifier_method = identifier_name

    def get_backdoor_variables(self, key=None):
        """ Return a list containing the backdoor variables.

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
        """Return a list containing the frontdoor variables (if present)
        """
        return self.frontdoor_variables

    def get_mediator_variables(self):
        """Return a list containing the mediator variables (if present)
        """
        return self.mediator_variables
    def get_instrumental_variables(self):
        """Return a list containing the instrumental variables (if present)
        """
        return self.instrumental_variables

    def __deepcopy__(self, memo):
        return IdentifiedEstimand(
                self.identifier, # not deep copied
                copy.deepcopy(self.treatment_variable),
                copy.deepcopy(self.outcome_variable),
                estimand_type=copy.deepcopy(self.estimand_type),
                estimands=copy.deepcopy(self.estimands),
                backdoor_variables=copy.deepcopy(self.backdoor_variables),
                instrumental_variables=copy.deepcopy(self.instrumental_variables),
                frontdoor_variables=copy.deepcopy(self.frontdoor_variables),
                mediator_variables=copy.deepcopy(self.mediator_variables),
                default_backdoor_id=copy.deepcopy(self.default_backdoor_id),
                identifier_method=copy.deepcopy(self.identifier_method)
            )

    def __str__(self, only_target_estimand=False, show_all_backdoor_sets=False):
        if self.no_directed_path:
            s = "No directed path from {0} to {1} in the causal graph.".format(
                    self.treatment_variable,
                    self.outcome_variable)
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
