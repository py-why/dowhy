""" Module containing the main model class for the dowhy package.

"""
import logging
from itertools import combinations

from sympy import init_printing

import dowhy.causal_estimators as causal_estimators
import dowhy.causal_refuters as causal_refuters
import dowhy.graph_learners as graph_learners
import dowhy.utils.cli_helpers as cli
from dowhy.causal_estimator import CausalEstimate, estimate_effect
from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment, IDIdentifier
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.causal_refuters.graph_refuter import GraphRefuter
from dowhy.utils.api import parse_state

init_printing()  # To display symbolic math symbols


class CausalModel:

    """Main class for storing the causal model state."""

    def __init__(
        self,
        data,
        treatment,
        outcome,
        graph=None,
        common_causes=None,
        instruments=None,
        effect_modifiers=None,
        estimand_type="nonparametric-ate",
        proceed_when_unidentifiable=False,
        missing_nodes_as_confounders=False,
        identify_vars=False,
        **kwargs,
    ):
        """Initialize data and create a causal graph instance.

        Assigns treatment and outcome variables.
        Also checks and finds the common causes and instruments for treatment
        and outcome.

        At least one of graph, common_causes or instruments must be provided. If
        none of these variables are provided, then learn_graph() can be used later.

        :param data: a pandas dataframe containing treatment, outcome and other
        variables.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param graph: path to DOT file containing a DAG or a string containing
        a DAG specification in DOT format
        :param common_causes: names of common causes of treatment and _outcome. Only used when graph is None.
        :param instruments: names of instrumental variables for the effect of
        treatment on outcome. Only used when graph is None.
        :param effect_modifiers: names of variables that can modify the treatment effect. If not provided, then the causal graph is used to find the effect modifiers. Estimators will return multiple different estimates based on each value of effect_modifiers.
        :param estimand_type: the type of estimand requested (currently only "nonparametric-ate" is supported). In the future, may support other specific parametric forms of identification.
        :param proceed_when_unidentifiable: does the identification proceed by ignoring potential unobserved confounders. Binary flag.
        :param missing_nodes_as_confounders: Binary flag indicating whether variables in the dataframe that are not included in the causal graph, should be  automatically included as confounder nodes.
        :param identify_vars: Variable deciding whether to compute common causes, instruments and effect modifiers while initializing the class. identify_vars should be set to False when user is providing common_causes, instruments or effect modifiers on their own(otherwise the identify_vars code can override the user provided values). Also it does not make sense if no graph is given.
        :returns: an instance of CausalModel class

        """
        self._data = data
        self._treatment = parse_state(treatment)
        self._outcome = parse_state(outcome)
        self._effect_modifiers = parse_state(effect_modifiers)
        self._estimand_type = estimand_type
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self._missing_nodes_as_confounders = missing_nodes_as_confounders
        self.logger = logging.getLogger(__name__)
        self._estimator_cache = {}

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
                    effect_modifier_names=self._effect_modifiers,
                    observed_node_names=self._data.columns.tolist(),
                )
            elif common_causes is not None:
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    common_cause_names=self._common_causes,
                    effect_modifier_names=self._effect_modifiers,
                    observed_node_names=self._data.columns.tolist(),
                )
            elif instruments is not None:
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    instrument_names=self._instruments,
                    effect_modifier_names=self._effect_modifiers,
                    observed_node_names=self._data.columns.tolist(),
                )
            else:
                self.logger.warning(
                    "Relevant variables to build causal graph not provided. You may want to use the learn_graph() function to construct the causal graph."
                )
                self._graph = CausalGraph(
                    self._treatment,
                    self._outcome,
                    effect_modifier_names=self._effect_modifiers,
                    observed_node_names=self._data.columns.tolist(),
                )
        else:
            self.init_graph(graph=graph, identify_vars=identify_vars)

        self._other_variables = kwargs
        self.summary()

    def init_graph(self, graph, identify_vars):
        """
        Initialize self._graph using graph provided by the user.

        """
        # Create causal graph object
        self._graph = CausalGraph(
            self._treatment,
            self._outcome,
            graph,
            effect_modifier_names=self._effect_modifiers,
            observed_node_names=self._data.columns.tolist(),
            missing_nodes_as_confounders=self._missing_nodes_as_confounders,
        )

        if identify_vars:
            self._common_causes = self._graph.get_common_causes(self._treatment, self._outcome)
            self._instruments = self._graph.get_instruments(self._treatment, self._outcome)
            # Sometimes, effect modifiers from the graph may not match those provided by the user.
            # (Because some effect modifiers may also be common causes)
            # In such cases, the user-provided modifiers are used.
            # If no effect modifiers are provided,  then the ones from the graph are used.
            if self._effect_modifiers is None or not self._effect_modifiers:
                self._effect_modifiers = self._graph.get_effect_modifiers(self._treatment, self._outcome)

    def get_common_causes(self):
        self._common_causes = self._graph.get_common_causes(self._treatment, self._outcome)
        return self._common_causes

    def get_instruments(self):
        self._instruments = self._graph.get_instruments(self._treatment, self._outcome)
        return self._instruments

    def get_effect_modifiers(self):
        self._effect_modifiers = self._graph.get_effect_modifiers(self._treatment, self._outcome)
        return self._effect_modifiers

    def learn_graph(self, method_name="cdt.causality.graph.LiNGAM", *args, **kwargs):
        """
        Learn causal graph from the data. This function takes the method name as input and initializes the
        causal graph object using the learnt graph.

        :param self: instance of the CausalModel class (or its subclass)
        :param method_name: Exact method name of the object to be imported from the concerned library.
        :returns: an instance of the CausalGraph class initialized with the learned graph.
        """
        # Import causal discovery class
        str_arr = method_name.split(".", maxsplit=1)
        library_name = str_arr[0]
        causal_discovery_class = graph_learners.get_discovery_class_object(library_name)

        model = causal_discovery_class(self._data, method_name, *args, **kwargs)
        graph = model.learn_graph()

        # Initialize causal graph object
        self.init_graph(graph=graph)

        return self._graph

    def identify_effect(
        self, estimand_type=None, method_name="default", proceed_when_unidentifiable=None, optimize_backdoor=False
    ):
        """Identify the causal effect to be estimated, using properties of the causal graph.

        :param method_name: Method name for identification algorithm. ("id-algorithm" or "default")
        :param proceed_when_unidentifiable: Binary flag indicating whether identification should proceed in the presence of (potential) unobserved confounders.
        :returns: a probability expression (estimand) for the causal effect if identified, else NULL

        """
        if proceed_when_unidentifiable is None:
            proceed_when_unidentifiable = self._proceed_when_unidentifiable
        if estimand_type is None:
            estimand_type = self._estimand_type

        estimand_type = EstimandType(estimand_type)

        if method_name == "id-algorithm":
            identifier = IDIdentifier()
        else:
            identifier = AutoIdentifier(
                estimand_type=estimand_type,
                backdoor_adjustment=BackdoorAdjustment(method_name),
                proceed_when_unidentifiable=proceed_when_unidentifiable,
                optimize_backdoor=optimize_backdoor,
            )

        identified_estimand = identifier.identify_effect(
            graph=self._graph, treatment_name=self._treatment, outcome_name=self._outcome
        )

        self.identifier = identifier

        return identified_estimand

    def estimate_effect(
        self,
        identified_estimand,
        method_name=None,
        control_value=0,
        treatment_value=1,
        test_significance=None,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        target_units="ate",
        effect_modifiers=None,
        fit_estimator=True,
        method_params=None,
    ):
        """Estimate the identified causal effect.

        Currently requires an explicit method name to be specified. Method names follow the convention of identification method followed by the specific estimation method: "[backdoor/iv].estimation_method_name". Following methods are supported.
            * Propensity Score Matching: "backdoor.propensity_score_matching"
            * Propensity Score Stratification: "backdoor.propensity_score_stratification"
            * Propensity Score-based Inverse Weighting: "backdoor.propensity_score_weighting"
            * Linear Regression: "backdoor.linear_regression"
            * Generalized Linear Models (e.g., logistic regression): "backdoor.generalized_linear_model"
            * Instrumental Variables: "iv.instrumental_variable"
            * Regression Discontinuity: "iv.regression_discontinuity"

        In addition, you can directly call any of the EconML estimation methods. The convention is "backdoor.econml.path-to-estimator-class". For example, for the double machine learning estimator ("DML" class) that is located inside "dml" module of EconML, you can use the method name, "backdoor.econml.dml.DML". CausalML estimators can also be called. See `this demo notebook <https://py-why.github.io/dowhy/example_notebooks/dowhy-conditional-treatment-effects.html>`_.


        :param identified_estimand: a probability expression
            that represents the effect to be estimated. Output of
            CausalModel.identify_effect method
        :param method_name: name of the estimation method to be used.
        :param control_value: Value of the treatment in the control group, for effect estimation.  If treatment is multi-variate, this can be a list.
        :param treatment_value: Value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list.
        :param test_significance: Binary flag on whether to additionally do a statistical signficance test for the estimate.
        :param evaluate_effect_strength: (Experimental) Binary flag on whether to estimate the relative strength of the treatment's effect. This measure can be used to compare different treatments for the same outcome (by running this method with different treatments sequentially).
        :param confidence_intervals: (Experimental) Binary flag indicating whether confidence intervals should be computed.
        :param target_units: (Experimental) The units for which the treatment effect should be estimated. This can be of three types. (1) a string for common specifications of target units (namely, "ate", "att" and "atc"), (2) a lambda function that can be used as an index for the data (pandas DataFrame), or (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
        :param effect_modifiers: Names of effect modifier variables can be (optionally) specified here too, since they do not affect identification. If None, the effect_modifiers from the CausalModel are used.
        :param fit_estimator: Boolean flag on whether to fit the estimator.
            Setting it to False is useful to estimate the effect on new data using a previously fitted estimator.
        :param method_params: Dictionary containing any method-specific parameters. These are passed directly to the estimating method. See the docs for each estimation method for allowed method-specific params.
        :returns: An instance of the CausalEstimate class, containing the causal effect estimate
            and other method-dependent information

        """

        if effect_modifiers is None or len(effect_modifiers) == 0:
            effect_modifiers = self._graph.get_effect_modifiers(self._treatment, self._outcome)

        if method_name is None:
            # TODO add propensity score as default backdoor method, iv as default iv method, add an informational message to show which method has been selected.
            pass
        else:
            # TODO add dowhy as a prefix to all dowhy estimators
            num_components = len(method_name.split("."))
            str_arr = method_name.split(".", maxsplit=1)
            identifier_name = str_arr[0]
            estimator_name = str_arr[1]
            # This is done as all dowhy estimators have two parts and external ones have two or more parts
            if num_components > 2:
                estimator_package = estimator_name.split(".")[0]
                if estimator_package == "dowhy":  # For updated dowhy methods
                    estimator_method = estimator_name.split(".", maxsplit=1)[
                        1
                    ]  # discard dowhy from the full package name
                    causal_estimator_class = causal_estimators.get_class_object(estimator_method + "_estimator")
                else:
                    third_party_estimator_package = estimator_package
                    causal_estimator_class = causal_estimators.get_class_object(
                        third_party_estimator_package, estimator_name
                    )
                    if method_params is None:
                        method_params = {}
                    # Define the third-party estimation method to be used
                    method_params[third_party_estimator_package + "_methodname"] = estimator_name
            else:  # For older dowhy methods
                self.logger.info(estimator_name)
                # Process the dowhy estimators
                causal_estimator_class = causal_estimators.get_class_object(estimator_name + "_estimator")

            if method_params is not None and (num_components <= 2 or estimator_package == "dowhy"):
                extra_args = method_params.get("init_params", {})
            else:
                extra_args = {}
            if method_params is None:
                method_params = {}

            identified_estimand.set_identifier_method(identifier_name)

            if not fit_estimator and method_name in self._estimator_cache:
                causal_estimator = self._estimator_cache[method_name]
            else:
                causal_estimator = causal_estimator_class(
                    self._data,
                    identified_estimand,
                    self._treatment,
                    self._outcome,  # names of treatment and outcome
                    control_value=control_value,
                    treatment_value=treatment_value,
                    test_significance=test_significance,
                    evaluate_effect_strength=evaluate_effect_strength,
                    confidence_intervals=confidence_intervals,
                    target_units=target_units,
                    effect_modifiers=effect_modifiers,
                    **method_params,
                    **extra_args,
                )
                self._estimator_cache[method_name] = causal_estimator

        return estimate_effect(
            self._treatment,
            self._outcome,
            identified_estimand,
            identifier_name,
            causal_estimator,
            control_value,
            treatment_value,
            test_significance,
            evaluate_effect_strength,
            confidence_intervals,
            target_units,
            effect_modifiers,
            fit_estimator,
            method_params,
        )

    def do(self, x, identified_estimand, method_name=None, fit_estimator=True, method_params=None):
        """Do operator for estimating values of the outcome after intervening on treatment.

        :param x: interventional value of the treatment variable
        :param identified_estimand: a probability expression
            that represents the effect to be estimated. Output of
            CausalModel.identify_effect method
        :param method_name: any of the estimation method to be used. See docs
            for estimate_effect method for a list of supported estimation methods.
        :param fit_estimator: Boolean flag on whether to fit the estimator.
            Setting it to False is useful to compute the do-operation on new
            data using a previously fitted estimator.
        :param method_params: Dictionary containing any method-specific parameters. These are passed directly to the estimating method.

        :returns: an instance of the CausalEstimate class, containing the causal effect estimate
            and other method-dependent information

        """
        if method_name is None:
            pass
        else:
            str_arr = method_name.split(".", maxsplit=1)
            identifier_name = str_arr[0]
            estimator_name = str_arr[1]
            identified_estimand.set_identifier_method(identifier_name)
            causal_estimator_class = causal_estimators.get_class_object(estimator_name + "_estimator")

        # Check if estimator's target estimand is identified
        if identified_estimand.estimands[identifier_name] is None:
            self.logger.warning("No valid identified estimand for using instrumental variables method")
            estimate = CausalEstimate(None, None, None, None, None)
        else:
            if fit_estimator:
                # Note that while the name of the variable is the same,
                # "self.causal_estimator", this estimator takes in less
                # parameters than the same from the
                # estimate_effect code. It is not advisable to use the
                # estimator from this function to call estimate_effect
                # with fit_estimator=False.
                self.causal_estimator = causal_estimator_class(
                    self._data,
                    identified_estimand,
                    self._treatment,
                    self._outcome,
                    test_significance=False,
                    **method_params,
                )
            else:
                # Estimator had been computed in a previous call
                assert self.causal_estimator is not None
            try:
                estimate = self.causal_estimator.do(x)
            except NotImplementedError:
                self.logger.error("Do Operation not implemented or not supported for this estimator.")
                raise NotImplementedError
        return estimate

    def refute_estimate(self, estimand, estimate, method_name=None, show_progress_bar=False, **kwargs):
        """Refute an estimated causal effect.

        If method_name is provided, uses the provided method. In the future, we may support automatic selection of suitable refutation tests. Following refutation methods are supported.
            * Adding a randomly-generated confounder: "random_common_cause"
            * Adding a confounder that is associated with both treatment and outcome: "add_unobserved_common_cause"
            * Replacing the treatment with a placebo (random) variable): "placebo_treatment_refuter"
            * Removing a random subset of the data: "data_subset_refuter"

        :param estimand: target estimand, an instance of the IdentifiedEstimand class (typically, the output of identify_effect)
        :param estimate: estimate to be refuted, an instance of the CausalEstimate class (typically, the output of estimate_effect)
        :param method_name: name of the refutation method
        :param show_progress_bar: Boolean flag on whether to show a progress bar
        :param kwargs:  (optional) additional arguments that are passed directly to the refutation method. Can specify a random seed here to ensure reproducible results ('random_seed' parameter). For method-specific parameters, consult the documentation for the specific method. All refutation methods are in the causal_refuters subpackage.

        :returns: an instance of the RefuteResult class

        """
        if estimate is None or estimate.value is None:
            self.logger.error("Aborting refutation! No estimate is provided.")
            raise ValueError("Aborting refutation! No valid estimate is provided.")
        if method_name is None:
            pass
        else:
            refuter_class = causal_refuters.get_class_object(method_name)

        refuter = refuter_class(self._data, identified_estimand=estimand, estimate=estimate, **kwargs)
        res = refuter.refute_estimate(show_progress_bar)
        return res

    def view_model(self, layout="dot", size=(8, 6), file_name="causal_model"):
        """View the causal DAG.

        :param layout: string specifying the layout of the graph.
        :param size: tuple (x, y) specifying the width and height of the figure in inches.
        :param file_name: string specifying the file name for the saved causal graph png.

        :returns: a visualization of the graph

        """
        self._graph.view_graph(layout, size, file_name)

    def interpret(self, method_name=None, **kwargs):
        """Interpret the causal model.

        :param method_name: method used for interpreting the model. If None,
                            then default interpreter is chosen that describes the model summary and shows the associated causal graph.
        :param kwargs:: Optional parameters that are directly passed to the interpreter method.

        :returns: None

        """
        if method_name is None:
            self.summary(print_to_stdout=True)
            self.view_model()
            return

        method_name_arr = parse_state(method_name)
        import dowhy.interpreters as interpreters

        for method in method_name_arr:
            interpreter = interpreters.get_class_object(method)
            interpreter(self, **kwargs).interpret()

    def summary(self, print_to_stdout=False):
        """Print a text summary of the model.

        :returns: a string containining the summary

        """
        summary_text = "Model to find the causal effect of treatment {0} on outcome {1}".format(
            self._treatment, self._outcome
        )
        self.logger.info(summary_text)
        if print_to_stdout:
            print(summary_text)
        return summary_text

    def refute_graph(self, k=1, independence_test=None, independence_constraints=None):
        """
        Check if the dependencies in input graph matches with the dataset -
        ( X ⫫ Y ) | Z
        where X and Y are considered as singleton sets currently
        Z can have multiple variables
        :param k: number of covariates in set Z
        :param independence_test: dictionary containing methods to test conditional independece in data
        :param independence_constraints: list of implications to be test input by the user in the format
            [(x,y,(z1,z2)),
            (x,y, (z3,))
            ]
        : returns: an instance of GraphRefuter class
        """
        if independence_test is not None:
            test_for_continuous = independence_test["test_for_continuous"]
            test_for_discrete = independence_test["test_for_discrete"]
            refuter = GraphRefuter(
                data=self._data, method_name_continuous=test_for_continuous, method_name_discrete=test_for_discrete
            )

        else:
            refuter = GraphRefuter(data=self._data)

        if independence_constraints is None:
            all_nodes = list(self._graph.get_all_nodes(include_unobserved=False))
            num_nodes = len(all_nodes)
            array_indices = list(range(0, num_nodes))
            all_possible_combinations = list(
                combinations(array_indices, 2)
            )  # Generating sets of indices of size 2 for different x and y
            conditional_independences = []
            self.logger.info("The followed conditional independences are true for the input graph")
            for combination in all_possible_combinations:  # Iterate over the unique 2-sized sets [x,y]
                i = combination[0]
                j = combination[1]
                a = all_nodes[i]
                b = all_nodes[j]
                if i < j:
                    temp_arr = all_nodes[:i] + all_nodes[i + 1 : j] + all_nodes[j + 1 :]
                else:
                    temp_arr = all_nodes[:j] + all_nodes[j + 1 : i] + all_nodes[i + 1 :]
                k_sized_lists = list(combinations(temp_arr, k))
                for k_list in k_sized_lists:
                    if self._graph.check_dseparation([str(a)], [str(b)], k_list) == True:
                        self.logger.info(" %s and %s are CI given %s ", a, b, k_list)
                        conditional_independences.append([a, b, k_list])

            independence_constraints = conditional_independences

        res = refuter.refute_model(independence_constraints=independence_constraints)

        self.logger.info(refuter._refutation_passed)

        return res
