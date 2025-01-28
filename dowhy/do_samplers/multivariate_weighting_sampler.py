from typing import List

import networkx as nx

from dowhy import EstimandType
from dowhy.do_sampler import DoSampler
from dowhy.utils.propensity_score import state_propensity_score


class MultivariateWeightingSampler(DoSampler):
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
        g, df, data_types

        """
        super().__init__(
            graph=graph,
            action_nodes=action_nodes,
            outcome_nodes=outcome_nodes,
            observed_nodes=observed_nodes,
            data=data,
            params=params,
            variable_types=variable_types,
            num_cores=num_cores,
            keep_original_treatment=keep_original_treatment,
            estimand_type=estimand_type,
        )

        self.logger.info("Using MultivariateWeightingSampler for do sampling.")
        self.logger.info("Caution: do samplers assume iid data.")

        self.point_sampler = False

    def make_treatment_effective(self, x):
        to_sample = self._df.copy()
        if not self.keep_original_treatment:
            for treatment, value in x.items():
                to_sample = to_sample[to_sample[treatment] == value]
        self._df = to_sample

    def disrupt_causes(self):
        self._df["state_propensity"] = state_propensity_score(
            self._data,
            self._target_estimand.get_adjustment_set(),
            self._treatment_names,
            variable_types=self._variable_types,
        )
        self._df["weight"] = self.compute_weights()

    def sample(self):
        self._df = self._df.sample(len(self._data), replace=True, weights=self._df["weight"])
        self._df.index = self._data.index

    def compute_weights(self):
        weights = 1.0 / self._df["state_propensity"]
        return weights
