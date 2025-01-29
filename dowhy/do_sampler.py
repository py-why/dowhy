import logging
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import EstimandType, identify_effect_auto
from dowhy.utils.api import parse_state


class DoSampler:
    """Base class for a sampler from the interventional distribution."""

    def __init__(
        self,
        graph: nx.DiGraph,
        action_nodes: List[str],
        outcome_nodes: List[str],
        observed_nodes: List[str],
        data,
        params=None,
        variable_types=None,
        num_cores=1,
        keep_original_treatment=False,
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    ):
        """
        Initializes a do sampler with data and names of relevant variables.

        Do sampling implements the do() operation from Pearl (2000). This is an operation is defined on a causal
        bayesian network, an explicit implementation of which is the basis for the MCMC sampling method.

        We abstract the idea behind the three-step process to allow other methods, as well. The `disrupt_causes`
        method is the means to make treatment assignment ignorable. In the Pearlian framework, this is where we cut the
        edges pointing into the causal state. With other methods, this will typically be by using some approach which
        assumes conditional ignorability (e.g. weighting, or explicit conditioning with Robins G-formula.)

        Next, the `make_treatment_effective` method reflects the assumption that the intervention we impose is
        "effective". Most simply, we fix the causal state to some specific value. We skip this step there is no value
        specified for the causal state, and the original values are used instead.

        Finally, we sample from the resulting distribution. This can be either from a `point_sample` method, in the case
        that the inference method doesn't support batch sampling, or the `sample` method in the case that it does. For
        convenience, the `point_sample` method parallelizes with `multiprocessing` using the `num_cores` kwargs to set
        the number of cores to use for parallelization.

        While different methods will have their own class attributes, the `_df` method should be common to all methods.
        This is them temporary dataset which starts as a copy of the original data, and is modified to reflect the steps
        of the do operation. Read through the existing methods (weighting is likely the most minimal) to get an idea of
        how this works to implement one yourself.

        :param data: pandas.DataFrame containing the data
        :param identified_estimand: dowhy.causal_identifier.IdentifiedEstimand: and estimand using a backdoor method
        for effect identification.
        :param treatments: list or str:  names of the treatment variables
        :param outcomes: list or str: names of the outcome variables
        :param variable_types: dict: A dictionary containing the variable's names and types. 'c' for continuous, 'o'
        for ordered, 'd' for discrete, and 'u' for unordered discrete.
        :param keep_original_treatment: bool: Whether to use `make_treatment_effective`, or to keep the original
        treatment assignments.
        :param params: (optional) additional method parameters

        """
        self._data = data.copy()
        self._target_estimand = identify_effect_auto(
            graph, action_nodes, outcome_nodes, observed_nodes, estimand_type=estimand_type
        )
        # TODO: Should this use the "general_adjustment" criterion instead?
        self._target_estimand.set_identifier_method("backdoor")
        self._treatment_names = parse_state(action_nodes)
        self._outcome_names = parse_state(outcome_nodes)
        self._estimate = None
        self._variable_types = variable_types
        self.num_cores = num_cores
        self.point_sampler = True
        self.sampler = None
        self.keep_original_treatment = keep_original_treatment

        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

        self._df = self._data.copy()

        if not self._variable_types:
            self._infer_variable_types()
        self.dep_type = [self._variable_types[var] for var in self._outcome_names]

        self.indep_type = [
            self._variable_types[var] for var in self._treatment_names + self._target_estimand.get_adjustment_set()
        ]
        self.density_types = [self._variable_types[var] for var in self._target_estimand.get_adjustment_set()]

        self.outcome_lower_support = self._data[self._outcome_names].min().values
        self.outcome_upper_support = self._data[self._outcome_names].max().values

        self.logger = logging.getLogger(__name__)

    def _sample_point(self, x_z):
        """
        Override this if your sampling method only allows sampling a point at a time.
        :param : numpy.array: x_z is a numpy array containing the values of x and z in the order of the list given by
        self._treatment_names + self._target_estimand.get_adjustment_set()
        :return: numpy.array:  a sampled outcome point
        """
        raise NotImplementedError

    def reset(self):
        """
        If your `DoSampler` has more attributes that the `_df` attribute, you should reset them all to their
        initialization values by overriding this method.
        :return:
        """
        self._df = self._data.copy()

    def make_treatment_effective(self, x):
        """
        This is more likely the implementation you'd like to use, but some methods may require overriding this method
        to make the treatment effective.
        :param x:
        :return:
        """
        if not self.keep_original_treatment:
            self._df[self._treatment_names] = x

    def disrupt_causes(self):
        """
        Override this method to render treatment assignment conditionally ignorable
        :return:
        """
        raise NotImplementedError

    def point_sample(self):
        if self.num_cores == 1:
            sampled_outcomes = self._df[self._treatment_names + self._target_estimand.get_adjustment_set()].apply(
                self._sample_point, axis=1
            )
        else:
            from multiprocessing import Pool

            p = Pool(self.num_cores)
            sampled_outcomes = np.array(
                p.map(
                    self.sampler.sample_point,
                    self._df[self._treatment_names + self._target_estimand.get_adjustment_set()].values,
                )
            )
            sampled_outcomes = pd.DataFrame(sampled_outcomes, columns=self._outcome_names)
        self._df[self._outcome_names] = sampled_outcomes

    def sample(self):
        """
        By default, this expects a sampler to be built on class initialization which contains a `sample` method.
        Override this method if you want to use a different approach to sampling.
        :return:
        """
        sampled_outcomes = self.sampler.sample(
            self._df[self._treatment_names + self._target_estimand.get_adjustment_set()].values
        )
        sampled_outcomes = pd.DataFrame(sampled_outcomes, columns=self._outcome_names)
        self._df[self._outcome_names] = sampled_outcomes

    def do_sample(self, x):
        self.reset()
        self.disrupt_causes()
        self.make_treatment_effective(x)
        if self.point_sampler:
            self.point_sample()
        else:
            self.sample()
        return self._df

    def _infer_variable_types(self):
        raise NotImplementedError("Variable type inference not implemented. Use the variable_types kwarg.")
