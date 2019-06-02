""" Module containing the main model class for the dowhy package.

"""

import logging

from sympy import init_printing

import dowhy.causal_estimators as causal_estimators
import dowhy.causal_refuters as causal_refuters
import dowhy.utils.cli_helpers as cli
from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import CausalIdentifier

from dowhy.utils.api import parse_state

init_printing()  # To display symbolic math symbols


class CausalModel:

    """Main class for storing the causal model state.

    """

    def __init__(self, data, treatment, outcome, graph=None,
                 common_causes=None, instruments=None, estimand_type="ate",
                 proceed_when_unidentifiable=False,
                 **kwargs):
        """Initialize data and create a causal graph instance.

        Assigns treatment and outcome variables.
        Also checks and finds the common causes and instruments for treatment
        and outcome.

        At least one of graph, common_causes or instruments must be provided.

        :param data: a pandas dataframe containing treatment, outcome and other
        variables.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param graph: path to DOT file containing a DAG or a string containing
        a DAG specification in DOT format
        :param common_causes: names of common causes of treatment and _outcome
        :param instruments: names of instrumental variables for the effect of
        treatment on outcome
        :returns: an instance of CausalModel class

        """
        self._data = data
        self._treatment = parse_state(treatment)
        self._outcome = parse_state(outcome)
        self._estimand_type = estimand_type
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        if 'logging_level' in kwargs:
            logging.basicConfig(level=kwargs['logging_level'])
        else:
            logging.basicConfig(level=logging.INFO)

        # TODO: move the logging level argument to a json file. Tue 20 Feb 2018 06:56:27 PM DST
        self.logger = logging.getLogger(__name__)

        if graph is None:
            self.logger.warning("Causal Graph not provided. DoWhy will construct a graph based on data inputs.")
            self._common_causes = parse_state(common_causes)
            self._instruments = parse_state(instruments)
            if common_causes is not None and instruments is not None:
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    common_cause_names=self._common_causes,
                    instrument_names=self._instruments,
                    observed_node_names=self._data.columns.tolist()
                )
            elif common_causes is not None:
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    common_cause_names=self._common_causes,
                    observed_node_names=self._data.columns.tolist()
                )
            elif instruments is not None:
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    instrument_names=self._instruments,
                    observed_node_names=self._data.columns.tolist()
                )
            else:
                cli.query_yes_no(
                    "WARN: Are you sure that there are no common causes of treatment and outcome?",
                    default=None
                )

        else:
            self._graph = CausalGraph(
                self._treatment,
                self._outcome,
                graph,
                observed_node_names=self._data.columns.tolist()
            )
            self._common_causes = self._graph.get_common_causes(self._treatment, self._outcome)
            self._instruments = self._graph.get_instruments(self._treatment,
                                                            self._outcome)

        self._other_variables = kwargs
        self.summary()

    def identify_effect(self, proceed_when_unidentifiable=None):
        """Identify the causal effect to be estimated, using properties of the causal graph.

        :returns: a probability expression for the causal effect if identified, else NULL

        """
        if proceed_when_unidentifiable is None:
            proceed_unidentifiable = self._proceed_when_unidentifiable

        self.identifier = CausalIdentifier(self._graph,
                                           self._estimand_type,
                                           proceed_when_unidentifiable=proceed_when_unidentifiable)
        identified_estimand = self.identifier.identify_effect()

        return identified_estimand

    def estimate_effect(self, identified_estimand, method_name=None,
                        test_significance=None, method_params=None):
        """Estimate the identified causal effect.

        If method_name is provided, uses the provided method. Else, finds a
        suitable method to be used.

        :param identified_estimand: a probability expression
            that represents the effect to be estimated. Output of
            CausalModel.identify_effect method
        :param method_name: (optional) name of the estimation method to be used.
        :returns: an instance of the CausalEstimate class, containing the causal effect estimate
            and other method-dependent information

        """
        if method_name is None:
            pass
        else:
            str_arr = method_name.split(".")
            identifier_name = str_arr[0]
            estimator_name = str_arr[1]
            identified_estimand.set_identifier_method(identifier_name)
            causal_estimator_class = causal_estimators.get_class_object(estimator_name + "_estimator")

        # Check if estimator's target estimand is identified
        if identified_estimand.estimands[identifier_name] is None:
            self.logger.warning("No valid identified estimand for using instrumental variables method")
            estimate = CausalEstimate(None, None, None)
        else:
            causal_estimator = causal_estimator_class(
                self._data,
                identified_estimand,
                self._treatment, self._outcome,
                test_significance=test_significance,
                params=method_params
            )
            estimate = causal_estimator.estimate_effect()
            estimate.add_params(
                estimand_type=identified_estimand.estimand_type,
                estimator_class=causal_estimator_class
            )
        return estimate

    def do(self, x, identified_estimand, method_name=None,  method_params=None):
        """Estimate the identified causal effect.

        If method_name is provided, uses the provided method. Else, finds a
        suitable method to be used.

        :param identified_estimand: a probability expression
            that represents the effect to be estimated. Output of
            CausalModel.identify_effect method
        :param method_name: (optional) name of the estimation method to be used.
        :returns: an instance of the CausalEstimate class, containing the causal effect estimate
            and other method-dependent information

        """
        if method_name is None:
            pass
        else:
            str_arr = method_name.split(".")
            identifier_name = str_arr[0]
            estimator_name = str_arr[1]
            identified_estimand.set_identifier_method(identifier_name)
            causal_estimator_class = causal_estimators.get_class_object(estimator_name + "_estimator")

        # Check if estimator's target estimand is identified
        if identified_estimand.estimands[identifier_name] is None:
            self.logger.warning("No valid identified estimand for using instrumental variables method")
            estimate = CausalEstimate(None, None, None)
        else:
            causal_estimator = causal_estimator_class(
                self._data,
                identified_estimand,
                self._treatment, self._outcome,
                test_significance=False,
                params=method_params
            )
            try:
                estimate = causal_estimator.do(x)
            except NotImplementedError:
                self.logger.error('Do Operation not implemented or not supported for this estimator.')
                raise NotImplementedError
        return estimate

    def refute_estimate(self, estimand, estimate, method_name=None, **kwargs):
        """Refute an estimated causal effect.

        If method_name is provided, uses the provided method. Else, finds a
        suitable method to use.

        :param estimate: an instance of the CausalEstimate class.
        :returns: an instance of the RefuteResult class

        """
        if method_name is None:
            pass
        else:
            refuter_class = causal_refuters.get_class_object(method_name)

        refuter = refuter_class(
            self._data,
            identified_estimand=estimand,
            estimate=estimate,
            **kwargs
        )
        res = refuter.refute_estimate()
        return res

    def view_model(self, layout="dot"):
        """View the causal DAG.

        :returns: a visualization of the graph

        """
        self._graph.view_graph(layout)

    def summary(self):
        """Print a text summary of the model.

        :returns: None

        """
        self.logger.info("Model to find the causal effect of treatment {0} on outcome {1}".format(self._treatment, self._outcome))
