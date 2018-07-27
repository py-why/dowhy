import logging

import networkx as nx
import pygraphviz as pgv


class CausalGraph:

    def __init__(self,
                 treatment_name, outcome_name,
                 graph=None,
                 common_cause_names=None,
                 instrument_names=None,
                 observed_node_names=None):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.fullname = "_".join([self.treatment_name,
                                  self.outcome_name,
                                  str(common_cause_names),
                                  str(instrument_names)])
        if graph is None:
            self._graph = pgv.AGraph(strict=True, directed=True)
            self._graph = self.build_graph(common_cause_names,
                                           instrument_names)
        else:
            self._graph = pgv.AGraph(graph, strict=True, directed=True)

        self._graph = nx.drawing.nx_agraph.from_agraph(self._graph)
        self._graph = self.add_node_attributes(observed_node_names)
        self._graph = self.add_unobserved_common_cause(observed_node_names)
        self.logger = logging.getLogger(__name__)

    def view_graph(self, layout="dot"):
        agraph = nx.drawing.nx_agraph.to_agraph(self._graph)
        agraph.draw("causal_model.png", format="png", prog=layout)

    def build_graph(self, common_cause_names, instrument_names):
        self._graph.add_node(self.treatment_name, observed="yes")
        self._graph.add_node(self.outcome_name, observed="yes")
        self._graph.add_edge(self.treatment_name, self.outcome_name)

        # Adding common causes
        if common_cause_names is not None:
            for node_name in common_cause_names:
                self._graph.add_node(node_name, observed="yes")
                self._graph.add_edge(node_name, self.treatment_name)
                self._graph.add_edge(node_name, self.outcome_name)
        # Adding instruments
        if instrument_names is not None:
            for node_name in instrument_names:
                self._graph.add_node(node_name, observed="yes")
                self._graph.add_edge(node_name, self.treatment_name)
        return self._graph

    def add_node_attributes(self, observed_node_names):
        for node_name in self._graph:
            if node_name in observed_node_names:
                self._graph.nodes[node_name]["observed"] = "yes"
            else:
                self._graph.nodes[node_name]["observed"] = "no"
        return self._graph

    def add_unobserved_common_cause(self, observed_node_names):
        # Adding unobserved confounders
        current_common_causes = self.get_common_causes(self.treatment_name,
                                                       self.outcome_name)
        create_new_common_cause = True
        for node_name in current_common_causes:
            if self._graph.nodes[node_name]["observed"] == "no":
                create_new_common_cause = False

        if create_new_common_cause:
            uc_label = "Unobserved Confounders"
            self._graph.add_node('U', label=uc_label)
            self._graph.add_edge('U', self.treatment_name)
            self._graph.add_edge('U', self.outcome_name)
        return self._graph

    def do_surgery(self, node_names, remove_outgoing_edges=False,
                   remove_incoming_edges=False):
        new_graph = self._graph.copy()
        for node_name in node_names:
            if remove_outgoing_edges:
                children = new_graph.successors(node_name)
                edges_bunch = [(node_name, child) for child in children]
                new_graph.remove_edges_from(edges_bunch)
            if remove_incoming_edges:
                parents = new_graph.predecessors(node_name)
                edges_bunch = [(parent, node_name) for parent in parents]
                new_graph.remove_edges_from(edges_bunch)
        return new_graph

    def get_common_causes(self, node1, node2):
        causes_node1 = self.get_ancestors(node1)
        causes_node2 = self.get_ancestors(node2)
        return list(causes_node1.intersection(causes_node2))

    def get_parents(self, node_name):
        return set(self._graph.predecessors(node_name))

    def get_ancestors(self, node_name):
        return set(nx.ancestors(self._graph, node_name))

    def get_descendants(self, node_name):
        return set(nx.descendants(self._graph, node_name))

    def all_observed(self, node_names):
        agraph = nx.drawing.nx_agraph.to_agraph(self._graph)
        for node_name in node_names:
            if agraph.get_node(node_name).attr["observed"] != "yes":
                return False

        return True

    def filter_unobserved_variables(self, node_names):
        observed_node_names = list()
        agraph = nx.drawing.nx_agraph.to_agraph(self._graph)
        for node_name in node_names:
            if agraph.get_node(node_name).attr["observed"] == "yes":
                observed_node_names.append(node_name)

        return observed_node_names

    def get_instruments(self, treatment_node, outcome_node):
        parents_treatment = self.get_parents(treatment_node)
        g_no_parents_treatment = self.do_surgery([treatment_node, ],
                                                 remove_incoming_edges=True)
        ancestors_outcome = nx.ancestors(g_no_parents_treatment, outcome_node)

        # Exclusion
        candidate_instruments = parents_treatment.difference(ancestors_outcome)
        self.logger.debug("Candidate instruments after exclusion %s",
                          candidate_instruments)
        # As-if-random setup
        children_causes_outcome = [nx.descendants(g_no_parents_treatment, v)
                                   for v in ancestors_outcome]
        children_causes_outcome = set([item
                                       for sublist in children_causes_outcome
                                       for item in sublist])

        # As-if-random
        instruments = candidate_instruments.difference(children_causes_outcome)
        return list(instruments)
