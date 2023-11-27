import networkx as nx
import pandas as pd

import dowhy.do_samplers as do_samplers
from dowhy import EstimandType
from dowhy.graph import build_graph
from dowhy.utils.api import parse_state


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        """
        An accessor for the pandas.DataFrame under the `causal` namespace.

        :param pandas_obj:
        """
        self._obj = pandas_obj
        self._graph = None
        self._sampler = None
        self._identified_estimand = None
        self._method = None

    def reset(self):
        """
        If a `causal` namespace method (especially `do`) was run statefully, this resets the namespace.

        :return:
        """
        self._graph = None
        self._identified_estimand = None
        self._sampler = None
        self._method = None

    def do(
        self,
        x,
        method="weighting",
        num_cores=1,
        variable_types={},
        outcome=None,
        params=None,
        graph: nx.DiGraph = None,
        common_causes=None,
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        stateful=False,
    ):
        """
        The do-operation implemented with sampling. This will return a pandas.DataFrame with the outcome
        variable(s) replaced with samples from P(Y|do(X=x)).

        If the value of `x` is left unspecified (e.g. as a string or list), then the original values of `x` are left in
        the DataFrame, and Y is sampled from its respective P(Y|do(x)). If the value of `x` is specified (passed with a
        `dict`, where variable names are keys, and values are specified) then the new `DataFrame` will contain the
        specified values of `x`.

        For some methods, the `variable_types` field must be specified. It should be a `dict`, where the keys are
        variable names, and values are 'o' for ordered discrete, 'u' for un-ordered discrete, 'd' for discrete, or 'c'
        for continuous.

        Inference requires a set of control variables. These can be provided explicitly using `common_causes`, which
        contains a list of variable names to control for. These can be provided implicitly by specifying a causal graph
        with `dot_graph`, from which they will be chosen using the default identification method.

        When the set of control variables can't be identified with the provided assumptions, a prompt will raise to the
        user asking whether to proceed. To automatically over-ride the prompt, you can set the flag
        `proceed_when_unidentifiable` to `True`.

        Some methods build components during inference which are expensive. To retain those components for later
        inference (e.g. successive calls to `do` with different values of `x`), you can set the `stateful` flag to `True`.
        Be cautious about using the `do` operation statefully. State is set on the namespace, rather than the method, so
        can behave unpredictably. To reset the namespace and run statelessly again, you can call the `reset` method.

        :param x: str, list, dict: The causal state on which to intervene, and (optional) its interventional value(s).
        :param method: The inference method to use with the sampler. Currently, `'mcmc'`, `'weighting'`, and
            `'kernel_density'` are supported. The `mcmc` sampler requires `pymc3>=3.7`.
        :param num_cores: int: if the inference method only supports sampling a point at a time, this will parallelize
            sampling.
        :param variable_types: dict: The dictionary containing the variable types. Must contain the union of the causal
            state, control variables, and the outcome.
        :param outcome: str: The outcome variable.
        :param params: dict: extra parameters to set as attributes on the sampler object
        :param dot_graph: str: A string specifying the causal graph.
        :param common_causes: list: A list of strings containing the variable names to control for.
        :param estimand_type: str: 'nonparametric-ate' is the only one currently supported. Others may be added later, to allow for specific, parametric estimands.
        :param proceed_when_unidentifiable: bool: A flag to over-ride user prompts to proceed when effects aren't
            identifiable with the assumptions provided.
        :param stateful: bool: Whether to retain state. By default, the do operation is stateless.

        :return: pandas.DataFrame: A DataFrame containing the sampled outcome
        """
        x, keep_original_treatment = self.parse_x(x)
        outcome = parse_state(outcome)
        if not stateful or method != self._method:
            self.reset()

        if graph is None:
            graph = build_graph(
                action_nodes=[xi for xi in x.keys()],
                outcome_nodes=outcome,
                common_cause_nodes=common_causes,
                effect_modifier_nodes=None,
                instrument_nodes=None,
                mediator_nodes=None,
            )

        if not bool(variable_types):  # check if the variables dictionary is empty
            variable_types = dict(self._obj.dtypes)  # Convert the series containing data types to a dictionary
            for key in variable_types.keys():
                variable_types[key] = self.convert_to_custom_type(
                    variable_types[key].name
                )  # Obtain the custom type corrosponding to each data type

        elif len(self._obj.columns) > len(variable_types):
            all_variables = dict(self._obj.dtypes)
            for key in all_variables.keys():
                if key not in variable_types:
                    variable_types[key] = self.convert_to_custom_type(all_variables[key].name)

        elif len(self._obj.columns) < len(variable_types):
            raise Exception("Number of variables in the DataFrame is lesser than the variable_types dict")

        if not self._sampler:
            self._method = method
            do_sampler_class = do_samplers.get_class_object(method + "_sampler")
            self._sampler = do_sampler_class(
                graph,
                observed_nodes=list(graph.nodes()),
                action_nodes=[xi for xi in x.keys()],
                outcome_nodes=outcome,
                data=self._obj,
                params=params,
                variable_types=variable_types,
                num_cores=num_cores,
                keep_original_treatment=keep_original_treatment,
                estimand_type=estimand_type,
            )
        result = self._sampler.do_sample(x)
        if not stateful:
            self.reset()
        return result

    def convert_to_custom_type(self, input_type):
        """
        This function converts a DataFrame type to a custom type used within dowhy.
        We make use of the following mapping
        int -> 'c'
        float -> 'c'
        binary -> 'b'
        category -> 'd'
        Currently we have not added support for time.
        :param input_type: str: The datatype of a column within a DataFrame
        """
        if "int" in input_type:
            return "c"
        elif "float" in input_type:
            return "c"
        elif "bool" in input_type:
            return "b"
        elif "category" in input_type:
            return "d"
        else:
            raise Exception("{} format is not supported".format(input_type))

    def parse_x(self, x):
        if type(x) == str:
            return {x: None}, True
        if type(x) == list:
            return {xi: None for xi in x}, True
        if type(x) == dict:
            return x, False
        raise Exception("x format not recognized: {}".format(type(x)))
