from typing import List

import networkx as nx
import numpy as np
import pymc3 as pm

from dowhy import EstimandType
from dowhy.do_sampler import DoSampler


class McmcSampler(DoSampler):
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

        self.logger.info("Using McmcSampler for do sampling.")
        self.point_sampler = False
        self.sampler = self._construct_sampler()

        self.g = graph.subgraph(observed_nodes)
        g_fit = nx.DiGraph(self.g)
        _, self.fit_trace = self.fit_causal_model(g_fit, self._data, self._variable_types)

    def apply_data_types(self, g, data_types):
        for node in nx.topological_sort(g):
            g.nodes()[node]["variable_type"] = data_types[node]
        return g

    def apply_parents(self, g):
        for node in nx.topological_sort(g):
            if not g.nodes()[node].get("parent_names"):
                g.nodes()[node]["parent_names"] = [parent for parent, _ in g.in_edges(node)]
        return g

    def apply_parameters(self, g, df, initialization_trace=None):
        for node in nx.topological_sort(g):
            parent_names = g.nodes()[node]["parent_names"]
            if parent_names:
                if not initialization_trace:
                    sd = np.array([df[node].std()] + (df[node].std() / df[parent_names].std()).tolist())
                    mu = np.array([df[node].std()] + (df[node].std() / df[parent_names].std()).tolist())
                    node_sd = df[node].std()
                else:
                    node_sd = initialization_trace["{}_sd".format(node)].mean()
                    mu = initialization_trace["beta_{}".format(node)].mean(axis=0)
                    sd = initialization_trace["beta_{}".format(node)].std(axis=0)
                g.nodes()[node]["parameters"] = pm.Normal(
                    "beta_{}".format(node), mu=mu, sd=sd, shape=len(parent_names) + 1
                )
                g.nodes()[node]["sd"] = pm.Exponential("{}_sd".format(node), lam=node_sd)
        return g

    def build_bayesian_network(self, g, df):
        for node in nx.topological_sort(g):
            if g.nodes()[node]["parent_names"]:
                mu = g.nodes()[node]["parameters"][0]  # intercept
                mu += pm.math.dot(df[g.nodes()[node]["parent_names"]], g.nodes()[node]["parameters"][1:])
                if g.nodes()[node]["variable_type"] == "c":
                    sd = g.nodes()[node]["sd"]
                    g.nodes()[node]["variable"] = pm.Normal("{}".format(node), mu=mu, sd=sd, observed=df[node])
                elif g.nodes()[node]["variable_type"] == "b":
                    g.nodes()[node]["variable"] = pm.Bernoulli("{}".format(node), logit_p=mu, observed=df[node])
                else:
                    raise Exception("Unrecognized variable type: {}".format(g.nodes()[node]["variable_type"]))
        return g

    def fit_causal_model(self, g, df, data_types, initialization_trace=None):
        if nx.is_directed_acyclic_graph(g):
            with pm.Model() as model:
                g = self.apply_data_types(g, data_types)
                g = self.apply_parents(g)
                g = self.apply_parameters(g, df, initialization_trace=initialization_trace)
                g = self.build_bayesian_network(g, df)
                trace = pm.sample(1000, tune=1000)
        else:
            raise Exception("Graph is not a DAG!")
        return g, trace

    def sample_prior_causal_model(self, g, df, data_types, initialization_trace):
        if nx.is_directed_acyclic_graph(g):
            with pm.Model() as model:
                g = self.apply_data_types(g, data_types)
                g = self.apply_parents(g)
                g = self.apply_parameters(g, df, initialization_trace=initialization_trace)
                g = self.build_bayesian_network(g, df)
                trace = pm.sample_prior_predictive(1)
        else:
            raise Exception("Graph is not a DAG!")
        return g, trace

    def do_x_surgery(self, g, x):
        for xi in x.keys():
            g.remove_edges_from([(parent, child) for (parent, child) in g.in_edges(xi)])
            g.nodes()[xi]["parent_names"] = []
        return g

    def make_intervention_effective(self, x):
        if not self.keep_original_treatment:
            for k, v in x.items():
                self._df[k] = v
        return self._df

    def do_sample(self, x):
        self.reset()
        g_for_surgery = nx.DiGraph(self.g)
        g_modified = self.do_x_surgery(g_for_surgery, x)
        self._df = self.make_intervention_effective(x)
        g_modified, trace = self.sample_prior_causal_model(
            g_modified, self._df, self._variable_types, initialization_trace=self.fit_trace
        )
        for col in self._df:
            if col in trace and col not in self._treatment_names:
                self._df[col] = trace[col]
        return self._df.copy()

    def _construct_sampler(self):
        pass
