import logging
import re
import networkx as nx
from dowhy.utils.api import parse_state
import itertools


class CausalGraph:

    def __init__(self,
                 treatment_name, outcome_name,
                 graph=None,
                 common_cause_names=None,
                 instrument_names=None,
                 observed_node_names=None):
        self.treatment_name = parse_state(treatment_name)
        self.outcome_name = parse_state(outcome_name)
        instrument_names = parse_state(instrument_names)
        common_cause_names = parse_state(common_cause_names)
        self.fullname = "_".join(self.treatment_name +
                                 self.outcome_name +
                                 common_cause_names +
                                 instrument_names)
        if graph is None:
            self._graph = nx.DiGraph()
            self._graph = self.build_graph(common_cause_names,
                                           instrument_names)
        elif re.match(r".*\.dot", graph):
            # load dot file
            try:
                import pygraphviz as pgv
                self._graph = nx.DiGraph(nx.drawing.nx_agraph.read_dot(graph))
            except Exception as e:
                print("Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot...")
                try:
                    import pydot
                    self._graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(graph))
                except Exception as e:
                    print("Error: Pydot cannot be loaded. " + str(e))
                    raise e
        elif re.match(r".*\.gml", graph):
            self._graph = nx.DiGraph(nx.read_gml(graph))
        elif re.match(r".*graph\s*\{.*\}\s*", graph):
            try:
                import pygraphviz as pgv
                self._graph = pgv.AGraph(graph, strict=True, directed=True)
                self._graph = nx.drawing.nx_agraph.from_agraph(self._graph)
            except Exception as e:
                print("Error: Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot ...")
                try:
                    import pydot
                    P_list = pydot.graph_from_dot_data(graph)
                    self._graph = nx.drawing.nx_pydot.from_pydot(P_list[0])
                except Exception as e:
                    print("Error: Pydot cannot be loaded. " + str(e))
                    raise e
        elif re.match(".*graph\s*\[.*\]\s*", graph):
            self._graph = nx.DiGraph(nx.parse_gml(graph))
        else:
            print("Error: Please provide graph (as string or text file) in dot or gml format.")
            print("Error: Incorrect graph format")
            raise ValueError

        self._graph = self.add_node_attributes(observed_node_names)
        self._graph = self.add_unobserved_common_cause(observed_node_names)
        self.logger = logging.getLogger(__name__)

    def view_graph(self, layout="dot"):
        out_filename = "causal_model.png"
        try:
            import pygraphviz as pgv
            agraph = nx.drawing.nx_agraph.to_agraph(self._graph)
            agraph.draw(out_filename, format="png", prog=layout)
        except:
            print("Warning: Pygraphviz cannot be loaded. Check that graphviz and pygraphviz are installed.")
            print("Using Matplotlib for plotting")
            import matplotlib.pyplot as plt
            plt.clf()
            nx.draw_networkx(self._graph, pos=nx.shell_layout(self._graph))
            plt.axis('off')
            plt.savefig(out_filename)
            plt.draw()

    def build_graph(self, common_cause_names, instrument_names):
        for treatment in self.treatment_name:
            self._graph.add_node(treatment, observed="yes")
        for outcome in self.outcome_name:
            self._graph.add_node(outcome, observed="yes")
        for treatment, outcome in itertools.product(self.treatment_name, self.outcome_name):
            self._graph.add_edge(treatment, outcome)

        # Adding common causes
        if common_cause_names is not None:
            for node_name in common_cause_names:
                for treatment, outcome in itertools.product(self.treatment_name, self.outcome_name):
                    self._graph.add_node(node_name, observed="yes")
                    self._graph.add_edge(node_name, treatment)
                    self._graph.add_edge(node_name, outcome)
        # Adding instruments
        if instrument_names:
            if type(instrument_names[0]) != tuple:
                if len(self.treatment_name) > 1:
                    self.logger.info("Assuming Instrument points to all treatments! Use tuples for more granularity.")
                for instrument, treatment in itertools.product(instrument_names, self.treatment_name):
                    self._graph.add_node(instrument, observed="yes")
                    self._graph.add_edge(instrument, treatment)
            else:
                for instrument, treatment in itertools.product(instrument_names):
                    self._graph.add_node(instrument, observed="yes")
                    self._graph.add_edge(instrument, treatment)
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
        print(current_common_causes)
        create_new_common_cause = True
        for node_name in current_common_causes:
            print(self._graph.nodes[node_name]["observed"] )
            print(self._graph.nodes[node_name])
            if self._graph.nodes[node_name]["observed"] == "no":
                create_new_common_cause = False

        if create_new_common_cause:
            uc_label = "Unobserved Confounders"
            self._graph.add_node('U', label=uc_label, observed="no")
            for node in self.treatment_name + self.outcome_name:
                self._graph.add_edge('U', node)
        return self._graph

    def get_unconfounded_observed_subgraph(self):
        observed_nodes = [node for node in self._graph.nodes() if self._graph.nodes[node]["observed"] == "yes"]
        return self._graph.subgraph(observed_nodes)

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

    def get_common_causes(self, nodes1, nodes2):
        nodes1 = parse_state(nodes1)
        nodes2 = parse_state(nodes2)
        causes_1 = set()
        causes_2 = set()
        for node in nodes1:
            causes_1 = causes_1.union(self.get_ancestors(node))
        for node in nodes2:
            causes_2 = causes_2.union(self.get_ancestors(node))
        return list(causes_1.intersection(causes_2))

    def get_parents(self, node_name):
        return set(self._graph.predecessors(node_name))

    def get_ancestors(self, node_name):
        return set(nx.ancestors(self._graph, node_name))

    def get_descendants(self, node_name):
        return set(nx.descendants(self._graph, node_name))

    def all_observed(self, node_names):
        for node_name in node_names:
            print(self._graph.nodes[node_name])
            if self._graph.nodes[node_name]["observed"] != "yes":
                return False

        return True

    def filter_unobserved_variables(self, node_names):
        observed_node_names = list()
        for node_name in node_names:
            if self._graph.nodes[node_name]["observed"] == "yes":
                observed_node_names.append(node_name)

        return observed_node_names

    def get_instruments(self, treatment_nodes, outcome_nodes):
        treatment_nodes = parse_state(treatment_nodes)
        outcome_nodes = parse_state(outcome_nodes)
        parents_treatment = set()
        for node in treatment_nodes:
            parents_treatment = parents_treatment.union(self.get_parents(node))
        g_no_parents_treatment = self.do_surgery(treatment_nodes,
                                                 remove_incoming_edges=True)
        ancestors_outcome = set()
        for node in outcome_nodes:
            ancestors_outcome = ancestors_outcome.union(nx.ancestors(g_no_parents_treatment, node))

        # [TODO: double check these work with multivariate implementation:]
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
