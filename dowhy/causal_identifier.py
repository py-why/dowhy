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

    def __init__(self, graph, estimand_type, proceed_when_unidentifiable=False):
        self._graph = graph
        self.estimand_type = estimand_type
        self.treatment_name = graph.treatment_name
        self.outcome_name = graph.outcome_name
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self.logger = logging.getLogger(__name__)

    def identify_effect(self): 
        """Main method that returns an identified estimand (if one exists). 

        Uses both backdoor and instrumental variable methods to check if an identified estimand exists, based on the causal graph. 

        :param self: instance of the CausalEstimator class (or its subclass)
        :returns:  target estimand, an instance of the IdentifiedEstimand class
        """

        estimands_dict = {}
        causes_t = self._graph.get_causes(self.treatment_name)
        causes_y = self._graph.get_causes(self.outcome_name, remove_edges={'sources':self.treatment_name, 'targets':self.outcome_name})
        common_causes = list(causes_t.intersection(causes_y))
        self.logger.info("Common causes of treatment and outcome:" + str(common_causes))
        if self._graph.all_observed(common_causes):
            self.logger.info("All common causes are observed. Causal effect can be identified.")
        else:
            self.logger.warning("If this is observed data (not from a randomized experiment), there might always be missing confounders. Causal effect cannot be identified perfectly.")
            if self._proceed_when_unidentifiable:
                self.logger.info(
                    "Continuing by ignoring these unobserved confounders because proceed_when_unidentifiable flag is True."
                )
            else:
                cli.query_yes_no(
                    "WARN: Do you want to continue by ignoring any unobserved confounders? (use proceed_when_unidentifiable=True to disable this prompt)",
                    default=None
                )
        observed_common_causes = self._graph.filter_unobserved_variables(common_causes)
        observed_common_causes = list(observed_common_causes)

        backdoor_estimand_expr = self.construct_backdoor_estimand(
            self.estimand_type, self._graph.treatment_name,
            self._graph.outcome_name, observed_common_causes
        )

        self.logger.debug("Identified expression = " + str(backdoor_estimand_expr))
        estimands_dict["backdoor"] = backdoor_estimand_expr

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

        estimand = IdentifiedEstimand(
            treatment_variable=self._graph.treatment_name,
            outcome_variable=self._graph.outcome_name,
            estimand_type=self.estimand_type,
            estimands=estimands_dict,
            backdoor_variables=observed_common_causes,
            instrumental_variables=instrument_names
        )
        return estimand

    def construct_backdoor_estimand(self, estimand_type, treatment_name,
                                    outcome_name, common_causes):
        # TODO: outputs string for now, but ideally should do symbolic
        # expressions Mon 19 Feb 2018 04:54:17 PM DST
        # TODO Better support for multivariate treatments

        expr = None
        if estimand_type == "nonparametric-ate":
            outcome_name = outcome_name[0]
            num_expr_str = outcome_name
            if len(common_causes)>0:
                num_expr_str += "|" + ",".join(common_causes)
            expr = "d(" + num_expr_str + ")/d" + ",".join(treatment_name)
            sym_mu = sp.Symbol("mu")
            sym_sigma = sp.Symbol("sigma", positive=True)
            sym_outcome = spstats.Normal(num_expr_str, sym_mu, sym_sigma)
            # sym_common_causes = [sp.stats.Normal(common_cause, sym_mu, sym_sigma) for common_cause in common_causes]
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
        else:
            raise ValueError("Estimand type not supported. Supported estimand types are 'non-parametric-ate'.")

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand

    def construct_iv_estimand(self, estimand_type, treatment_name,
                              outcome_name, instrument_names):
        # TODO: support multivariate treatments better.
        expr = None
        if estimand_type == "nonparametric-ate":
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
        else:
            raise ValueError("Estimand type not supported. Supported estimand types are 'non-parametric-ate'.")

        estimand = {
            'estimand': sym_effect,
            'assumptions': sym_assumptions
        }
        return estimand


class IdentifiedEstimand:

    """Class for storing a causal estimand, typically as a result of the identification step. 

    """

    def __init__(self, treatment_variable, outcome_variable,
                 estimand_type=None, estimands=None,
                 backdoor_variables=None, instrumental_variables=None):
        self.treatment_variable = parse_state(treatment_variable)
        self.outcome_variable = parse_state(outcome_variable)
        self.backdoor_variables = parse_state(backdoor_variables)
        self.instrumental_variables = parse_state(instrumental_variables)
        self.estimand_type = estimand_type
        self.estimands = estimands
        self.identifier_method = None

    def set_identifier_method(self, identifier_name):
        self.identifier_method = identifier_name

    def __str__(self):
        s = "Estimand type: {0}\n".format(self.estimand_type)
        i = 1
        for k, v in self.estimands.items():
            s += "### Estimand : {0}\n".format(i)
            s += "Estimand name: {0}\n".format(k)
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
