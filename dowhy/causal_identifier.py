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
    VALID_METHOD_NAMES = {"default", "exhaustive-search"}

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

    def identify_effect(self):
        """Main method that returns an identified estimand (if one exists).

        If estimand_type is non-parametric ATE, then  uses backdoor, instrumental variable and frontdoor identification methods,  to check if an identified estimand exists, based on the causal graph.

        :param self: instance of the CausalEstimator class (or its subclass)
        :returns:  target estimand, an instance of the IdentifiedEstimand class
        """
        if self.estimand_type == CausalIdentifier.NONPARAMETRIC_ATE:
            return self.identify_ate_effect()
        elif self.estimand_type == CausalIdentifier.NONPARAMETRIC_NDE:
            return self.identify_nde_effect()
        elif self.estimand_type == CausalIdentifier.NONPARAMETRIC_NIE:
            return self.identify_nie_effect()
        else:
            raise ValueError("Estimand type is not supported. Use either {0}, {1}, or {2}.".format(
                CausalIdentifier.NONPARAMETRIC_ATE,
                CausalIdentifier.NONPARAMETRIC_NDE,
                CausalIdentifier.NONPARAMETRIC_NIE))

    def identify_ate_effect(self):
        estimands_dict = {}
        mediation_first_stage_confounders = None
        mediation_second_stage_confounders = None
        ### 1. BACKDOOR IDENTIFICATION
        # First, checking if there are any valid backdoor adjustment sets
        backdoor_sets = self.identify_backdoor(self.treatment_name, self.outcome_name)
        estimands_dict, backdoor_variables_dict = self.build_backdoor_estimands_dict(
                self.treatment_name,
                self.outcome_name,
                backdoor_sets,
                estimands_dict)
        # Setting default "backdoor" identification adjustment set
        default_backdoor_id = self.get_default_backdoor_set_id(backdoor_variables_dict)
        estimands_dict["backdoor"] = estimands_dict.get(str(default_backdoor_id), None)
        backdoor_variables_dict["backdoor"] = backdoor_variables_dict.get(str(default_backdoor_id), None)

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




    def identify_backdoor(self, treatment_name, outcome_name):
        backdoor_sets = []
        backdoor_paths = self._graph.get_backdoor_paths(treatment_name, outcome_name)
        # First, checking if empty set is a valid backdoor set
        empty_set = set()
        check = self._graph.check_valid_backdoor_set(treatment_name, outcome_name, empty_set,
                backdoor_paths=backdoor_paths)
        if check["is_dseparated"]:
            backdoor_sets.append({
                'backdoor_set':empty_set,
                'num_paths_blocked_by_observed_nodes': check["num_paths_blocked_by_observed_nodes"]})
        # Second, checking for all other sets of variables
        eligible_variables = self._graph.get_all_nodes() \
            - set(treatment_name) \
            - set(outcome_name) \
            - set(self._graph.get_instruments(treatment_name, outcome_name))
        eligible_variables -= self._graph.get_descendants(treatment_name)

        num_iterations = 0
        found_valid_adjustment_set = False
        if self.method_name in CausalIdentifier.VALID_METHOD_NAMES:
            for size_candidate_set in range(len(eligible_variables), 0, -1):
                for candidate_set in itertools.combinations(eligible_variables, size_candidate_set):
                    check = self._graph.check_valid_backdoor_set(treatment_name,
                            outcome_name, candidate_set, backdoor_paths=backdoor_paths)
                    self.logger.debug("Candidate backdoor set: {0}, is_dseparated: {1}, No. of paths blocked by observed_nodes: {2}".format(candidate_set, check["is_dseparated"], check["num_paths_blocked_by_observed_nodes"]))
                    if check["is_dseparated"]:
                        backdoor_sets.append({
                            'backdoor_set': candidate_set,
                            'num_paths_blocked_by_observed_nodes': check["num_paths_blocked_by_observed_nodes"]})
                        if self._graph.all_observed(candidate_set):
                            found_valid_adjustment_set = True
                    num_iterations += 1
                    if self.method_name == "default" and num_iterations > CausalIdentifier.MAX_BACKDOOR_ITERATIONS:
                        break
                if self.method_name == "default" and found_valid_adjustment_set:
                    break
        else:
            raise ValueError(f"Identifier method {self.method_name} not supported. Try one of the following: {CausalIdentifier.VALID_METHOD_NAMES}")
        #causes_t = self._graph.get_causes(self.treatment_name)
        #causes_y = self._graph.get_causes(self.outcome_name, remove_edges={'sources':self.treatment_name, 'targets':self.outcome_name})
        #common_causes = list(causes_t.intersection(causes_y))
        #self.logger.info("Common causes of treatment and outcome:" + str(common_causes))
        observed_backdoor_sets = [ bset for bset in backdoor_sets if self._graph.all_observed(bset["backdoor_set"])]
        if len(observed_backdoor_sets)==0:
            return backdoor_sets
        else:
            return observed_backdoor_sets

    def get_default_backdoor_set_id(self, backdoor_sets_dict):
        # Adding a None estimand if no backdoor set found
        if len(backdoor_sets_dict) == 0:
            return None
        max_set_length = -1
        default_key = None
        # Default set is the one with the most number of adjustment variables (optimizing for minimum (unknown) bias not for efficiency)
        for key, bdoor_set in backdoor_sets_dict.items():
            if len(bdoor_set) > max_set_length:
                max_set_length = len(bdoor_set)
                default_key = key
        return default_key

    def build_backdoor_estimands_dict(self, treatment_name, outcome_name,
            backdoor_sets, estimands_dict, proceed_when_unidentifiable=None):
        backdoor_variables_dict = {}
        if proceed_when_unidentifiable is None:
            proceed_when_unidentifiable = self._proceed_when_unidentifiable
        is_identified = [ self._graph.all_observed(bset["backdoor_set"]) for bset in backdoor_sets ]

        if all(is_identified):
            self.logger.info("All common causes are observed. Causal effect can be identified.")
            backdoor_sets_arr = [list(
                bset["backdoor_set"])
                for bset in backdoor_sets]
        else: # there is unobserved confounding
            self.logger.warning("If this is observed data (not from a randomized experiment), there might always be missing confounders. Causal effect cannot be identified perfectly.")
            response = False # user response
            if proceed_when_unidentifiable:
                self.logger.info(
                    "Continuing by ignoring these unobserved confounders because proceed_when_unidentifiable flag is True."
                )
            else:
                response= cli.query_yes_no(
                    "WARN: Do you want to continue by ignoring any unobserved confounders? (use proceed_when_unidentifiable=True to disable this prompt)",
                    default=None
                )
                if response is False:
                    self.logger.warn("Identification failed due to unobserved variables.")
                    backdoor_sets_arr = []
            if proceed_when_unidentifiable or response is True:
                max_paths_blocked = max( bset['num_paths_blocked_by_observed_nodes'] for bset in backdoor_sets)
                backdoor_sets_arr = [list(
                    self._graph.filter_unobserved_variables(bset["backdoor_set"]))
                    for bset in backdoor_sets
                    if bset["num_paths_blocked_by_observed_nodes"]==max_paths_blocked]

        for i in range(len(backdoor_sets_arr)):
            backdoor_estimand_expr = self.construct_backdoor_estimand(
                self.estimand_type, treatment_name,
                outcome_name, backdoor_sets_arr[i])
            self.logger.debug("Identified expression = " + str(backdoor_estimand_expr))
            estimands_dict["backdoor"+str(i+1)] = backdoor_estimand_expr
            backdoor_variables_dict["backdoor"+str(i+1)] = backdoor_sets_arr[i]
        return estimands_dict, backdoor_variables_dict

    def identify_frontdoor(self):
        """ Find a valid frontdoor variable if it exists.

        Currently only supports a single variable frontdoor set.
        """
        frontdoor_var = None
        frontdoor_paths = self._graph.get_all_directed_paths(self.treatment_name, self.outcome_name)
        eligible_variables = self._graph.get_descendants(self.treatment_name) \
            - set(self.outcome_name)
        # For simplicity, assuming a one-variable frontdoor set
        for candidate_var in eligible_variables:
            is_valid_frontdoor = self._graph.check_valid_frontdoor_set(self.treatment_name,
                    self.outcome_name, parse_state(candidate_var), frontdoor_paths=frontdoor_paths)
            self.logger.debug("Candidate frontdoor set: {0}, is_dseparated: {1}".format(candidate_var, is_valid_frontdoor))
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

    def get_default_backdoor_set_id(self, backdoor_sets_dict):
        # Adding a None estimand if no backdoor set found
        if len(backdoor_sets_dict) == 0:
            return None
        max_set_length = -1
        default_key = None
        # Default set is the one with the most number of adjustment variables (optimizing for minimum (unknown) bias not for efficiency)
        for key, bdoor_set in backdoor_sets_dict.items():
            if len(bdoor_set) > max_set_length:
                max_set_length = len(bdoor_set)
                default_key = key
        return default_key

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
                 default_backdoor_id=None, identifier_method=None):
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

    def set_identifier_method(self, identifier_name):
        self.identifier_method = identifier_name

    def get_backdoor_variables(self, key=None):
        """ Return a list containing the backdoor variables.

            If the calling estimator method is a backdoor method, return the
            backdoor variables corresponding to its target estimand.
            Otherwise, return the backdoor variables for the default backdoor estimand.
        """
        if key is None:
            if self.identifier_method.startswith("backdoor"):
                return self.backdoor_variables[self.identifier_method]
            else:
                return self.backdoor_variables[self.default_backdoor_id]
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

    def __str__(self, only_target_estimand=False):
        s = "Estimand type: {0}\n".format(self.estimand_type)
        i = 1
        has_valid_backdoor = sum("backdoor" in key for key in self.estimands.keys())
        for k, v in self.estimands.items():
            # Do not show backdoor key unless it is the only backdoor set.
            if k == "backdoor" and has_valid_backdoor > 1:
                continue
            if only_target_estimand and k != self.identifier_method:
                continue
            s += "\n### Estimand : {0}\n".format(i)
            s += "Estimand name: {0}".format(k)
            if k == self.default_backdoor_id:
                s += " (Default)"
            s += "\n"
            if v is None:
                s += "No such variable found!\n"
            else:
                sp_expr_str = sp.pretty(v["estimand"], use_unicode=True)
                s += "Estimand expression:\n{0}\n".format(sp_expr_str)
                j = 1
                for ass_name, ass_str in v["assumptions"].items():
                    s += "Estimand assumption {0}, {1}: {2}\n".format(j, ass_name, ass_str)
                    j += 1
            i += 1
        return s
