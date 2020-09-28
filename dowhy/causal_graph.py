import logging
import re
import networkx as nx
from dowhy.utils.api import parse_state
import itertools


class CausalGraph:

    """Class for creating and modifying the causal graph.

       Accepts a graph string (or a text file) in gml format (preferred) and dot format. Graphviz-like attributes can be set for edges and nodes. E.g. style="dashed" as an edge attribute ensures that the edge is drawn with a dashed line.

        If a graph string is not given, names of treatment, outcome, and confounders, instruments and effect modifiers (if any) can be provided to create the graph.
    """

    def __init__(self,
                 treatment_name, outcome_name,
                 graph=None,
                 common_cause_names=None,
                 instrument_names=None,
                 effect_modifier_names=None,
                 observed_node_names=None,
                 missing_nodes_as_confounders=False):
        self.treatment_name = parse_state(treatment_name)
        self.outcome_name = parse_state(outcome_name)
        instrument_names = parse_state(instrument_names)
        common_cause_names = parse_state(common_cause_names)
        self.logger = logging.getLogger(__name__)

        if graph is None:
            self._graph = nx.DiGraph()
            self._graph = self.build_graph(common_cause_names,
                                           instrument_names, effect_modifier_names)
        elif re.match(r".*\.dot", graph):
            # load dot file
            try:
                import pygraphviz as pgv
                self._graph = nx.DiGraph(nx.drawing.nx_agraph.read_dot(graph))
            except Exception as e:
                self.logger.error("Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot...")
                try:
                    import pydot
                    self._graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(graph))
                except Exception as e:
                    self.logger.error("Error: Pydot cannot be loaded. " + str(e))
                    raise e
        elif re.match(r".*\.gml", graph):
            self._graph = nx.DiGraph(nx.read_gml(graph))
        elif re.match(r".*graph\s*\{.*\}\s*", graph):
            try:
                import pygraphviz as pgv
                self._graph = pgv.AGraph(graph, strict=True, directed=True)
                self._graph = nx.drawing.nx_agraph.from_agraph(self._graph)
            except Exception as e:
                self.logger.error("Error: Pygraphviz cannot be loaded. " + str(e) + "\nTrying pydot ...")
                try:
                    import pydot
                    P_list = pydot.graph_from_dot_data(graph)
                    self._graph = nx.drawing.nx_pydot.from_pydot(P_list[0])
                except Exception as e:
                    self.logger.error("Error: Pydot cannot be loaded. " + str(e))
                    raise e
        elif re.match(".*graph\s*\[.*\]\s*", graph):
            self._graph = nx.DiGraph(nx.parse_gml(graph))
        else:
            self.logger.error("Error: Please provide graph (as string or text file) in dot or gml format.")
            self.logger.error("Error: Incorrect graph format")
            raise ValueError
        if missing_nodes_as_confounders:
            self._graph = self.add_missing_nodes_as_common_causes(observed_node_names)
        # Adding node attributes
        self._graph = self.add_node_attributes(observed_node_names)
        #TODO do not add it here. CausalIdentifier should call causal_graph to add an unobserved common cause if needed. This also ensures that we do not need get_common_causes in this class.
        self._graph = self.add_unobserved_common_cause(observed_node_names)

    def view_graph(self, layout="dot"):
        out_filename = "causal_model.png"
        try:
            import pygraphviz as pgv
            agraph = nx.drawing.nx_agraph.to_agraph(self._graph)
            agraph.draw(out_filename, format="png", prog=layout)
        except:
            self.logger.warning("Warning: Pygraphviz cannot be loaded. Check that graphviz and pygraphviz are installed.")
            self.logger.info("Using Matplotlib for plotting")
            import matplotlib.pyplot as plt

            solid_edges = [(n1,n2) for n1,n2, e in self._graph.edges(data=True) if 'style' not in e ]
            dashed_edges =[(n1,n2) for n1,n2, e in self._graph.edges(data=True) if ('style' in e and e['style']=="dashed") ]
            plt.clf()

            pos = nx.layout.shell_layout(self._graph)
            nx.draw_networkx_nodes(self._graph, pos, node_color='yellow',node_size=400 )
            nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=solid_edges,
                    arrowstyle="-|>",
                    arrowsize=12)
            nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=dashed_edges,
                    arrowstyle="-|>",
                    style="dashed",
                    arrowsize=12)
            
            labels = nx.draw_networkx_labels(self._graph, pos)

            plt.axis('off')
            plt.savefig(out_filename)
            plt.draw()

    def build_graph(self, common_cause_names, instrument_names, effect_modifier_names):
        """ Creates nodes and edges based on variable names and their semantics.
        
        Currently only considers the graphical representation of "direct" effect modifiers. Thus, all effect modifiers are assumed to be "direct" unless otherwise expressed using a graph. Based on the taxonomy of effect modifiers by VanderWheele and Robins: "Four types of effect modification: A classification based on directed acyclic graphs. Epidemiology. 2007."
        """

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

        # Adding effect modifiers
        if effect_modifier_names is not None:
            for node_name in effect_modifier_names:
                if node_name not in common_cause_names:
                    for outcome in self.outcome_name:
                        self._graph.add_node(node_name, observed="yes")
                        self._graph.add_edge(node_name, outcome, style = "dotted", headport="s", tailport="n")
                        self._graph.add_edge(outcome, node_name, style = "dotted", headport="n", tailport="s") # TODO make the ports more general so that they apply not just to top-bottom node configurations
        return self._graph

    def add_node_attributes(self, observed_node_names):
        for node_name in self._graph:
            if node_name in observed_node_names:
                self._graph.nodes[node_name]["observed"] = "yes"
            else:
                self._graph.nodes[node_name]["observed"] = "no"
        return self._graph

    def add_missing_nodes_as_common_causes(self, observed_node_names):
        # Adding columns in the dataframe as confounders that were not in the graph
        for node_name in observed_node_names:
            if node_name not in self._graph:
                self._graph.add_node(node_name, observed="yes")
                for treatment_outcome_node in self.treatment_name + self.outcome_name:
                    self._graph.add_edge(node_name, treatment_outcome_node)
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
            self._graph.add_node('U', label=uc_label, observed="no")
            for node in self.treatment_name + self.outcome_name:
                self._graph.add_edge('U', node)
            self.logger.info('If this is observed data (not from a randomized experiment), there might always be missing confounders. Adding a node named "Unobserved Confounders" to reflect this.')
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

    def get_causes(self, nodes, remove_edges = None):
        nodes = parse_state(nodes)
        new_graph=None
        if remove_edges is not None:
            new_graph = self._graph.copy()  # caution: shallow copy of the attributes
            sources = parse_state(remove_edges["sources"])
            targets = parse_state(remove_edges["targets"])
            for s in sources:
                for t in targets:
                    new_graph.remove_edge(s, t)
        causes = set()
        for v in nodes:
            causes = causes.union(self.get_ancestors(v, new_graph=new_graph))
        return causes

    def check_valid_backdoor_set(self, nodes1, nodes2, nodes3, backdoor_paths=None):
        # also return the number of backdoor paths blocked by observed nodes
        if backdoor_paths is None:
            backdoor_paths = self.get_backdoor_paths(nodes1, nodes2)
        d_separated = all([self.is_blocked(path, nodes3) for path in backdoor_paths])
        observed_nodes3 = self.filter_unobserved_variables(nodes3)
        num_paths_blocked = sum([self.is_blocked(path, observed_nodes3) for path in backdoor_paths])
        return {'is_dseparated': d_separated,
                'num_paths_blocked_by_observed_nodes': num_paths_blocked}

    def get_backdoor_paths(self, nodes1, nodes2):
        paths = []
        undirected_graph = self._graph.to_undirected()
        nodes12 = set(nodes1).union(nodes2)
        for node1 in nodes1:
            for node2 in nodes2:
                backdoor_paths = [
                        pth
                        for pth in nx.all_simple_paths(undirected_graph, source=node1, target=node2)
                        if self._graph.has_edge(pth[1], pth[0])]
                # remove paths that have nodes1\node1 or nodes2\node2 as intermediate nodes
                filtered_backdoor_paths = [
                        pth
                        for pth in backdoor_paths
                        if len(nodes12.intersection(pth[1:-1]))==0]
                paths.extend(filtered_backdoor_paths)
        self.logger.debug("Backdoor paths: " + str(paths))
        return paths

    def is_blocked(self, path, conditioned_nodes):
        """ Uses d-separation criteria to decide if conditioned_nodes block given path.
        """

        blocked_by_conditioning = False
        has_unconditioned_collider = False
        for i in range(len(path)-2):
            if self._graph.has_edge(path[i], path[i+1]) and self._graph.has_edge(path[i+2], path[i+1]): # collider
                collider_descendants = nx.descendants(self._graph, path[i+1])
                if path[i+1] not in conditioned_nodes and all(cdesc not in conditioned_nodes for cdesc in collider_descendants):
                    has_unconditioned_collider=True
            else: # chain or fork
                if path[i+1] in conditioned_nodes:
                    blocked_by_conditioning = True
                    break
        if blocked_by_conditioning:
            return True
        elif has_unconditioned_collider:
            return True
        else:
            return False

    def get_common_causes(self, nodes1, nodes2):
        """
        Assume that nodes1 causes nodes2 (e.g., nodes1 are the treatments and nodes2 are the outcomes)
        """
        # TODO Refactor to remove this from here and only implement this logic in causalIdentifier. Unnecessary assumption of nodes1 to be causing nodes2.
        nodes1 = parse_state(nodes1)
        nodes2 = parse_state(nodes2)
        causes_1 = set()
        causes_2 = set()
        for node in nodes1:
            causes_1 = causes_1.union(self.get_ancestors(node))
        for node in nodes2:
            # Cannot simply compute ancestors, since that will also include nodes1 and its parents (e.g. instruments)
            parents_2 = self.get_parents(node)
            for parent in parents_2:
                if parent not in nodes1:
                    causes_2 = causes_2.union(set([parent,]))
                    causes_2 = causes_2.union(self.get_ancestors(parent))
        return list(causes_1.intersection(causes_2))

    def get_effect_modifiers(self, nodes1, nodes2):
        modifiers = set()
        for node in nodes2:
            modifiers = modifiers.union(self.get_ancestors(node))
        modifiers = modifiers.difference(nodes1)
        for node in nodes1:
            modifiers = modifiers.difference(self.get_ancestors(node))
        # removing all mediators
        for node1 in nodes1:
            for node2 in nodes2:
                all_directed_paths = nx.all_simple_paths(self._graph, node1, node2)
                for path in all_directed_paths:
                    modifiers=modifiers.difference(path)
        return list(modifiers)

    def get_parents(self, node_name):
        return set(self._graph.predecessors(node_name))

    def get_ancestors(self, node_name, new_graph=None):
        if new_graph is None:
            graph=self._graph
        else:
            graph=new_graph
        return set(nx.ancestors(graph, node_name))

    def get_descendants(self, nodes):
        descendants = set()
        for node_name in nodes:
            descendants = descendants.union(set(nx.descendants(self._graph, node_name)))
        return descendants

    def all_observed(self, node_names):
        for node_name in node_names:
            if self._graph.nodes[node_name]["observed"] != "yes":
                return False

        return True

    def get_all_nodes(self, include_unobserved=True):
        return self._graph.nodes

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
        self.logger.debug("Candidate instruments after satisfying exclusion: %s",
                          candidate_instruments)
        # As-if-random setup
        children_causes_outcome = [nx.descendants(g_no_parents_treatment, v)
                                   for v in ancestors_outcome]
        children_causes_outcome = set([item
                                       for sublist in children_causes_outcome
                                       for item in sublist])

        # As-if-random
        instruments = candidate_instruments.difference(children_causes_outcome)
        self.logger.debug("Candidate instruments after satisfying exclusion and as-if-random: %s",
                instruments)
        return list(instruments)


    def get_all_directed_paths(self, nodes1, nodes2):
        """ Get all directed paths between sets of nodes.

        Currently only supports singleton sets.
        """
        node1 = nodes1[0]
        node2 = nodes2[0]
        return nx.all_simple_paths(self._graph, source=node1, target=node2)

    def check_valid_frontdoor_set(self, nodes1, nodes2, candidate_nodes, frontdoor_paths=None):
        """Check if valid the frontdoor variables for set of treatments, nodes1 to set of outcomes, nodes2.
        """
        if frontdoor_paths is None:
            frontdoor_paths = self.get_all_directed_paths(nodes1, nodes2)

        d_separated = all([self.is_blocked(path, candidate_nodes) for path in frontdoor_paths])
        return d_separated

    def check_valid_mediation_set(self, nodes1, nodes2, candidate_nodes, mediation_paths=None):
        """Check if candidate nodes are valid mediators for set of treatments, nodes1 to set of outcomes, nodes2.
        """
        if mediation_paths is None:
            mediation_paths = self.get_all_directed_paths(nodes1, nodes2)

        is_mediator = any([self.is_blocked(path, candidate_nodes) for path in mediation_paths])
        return is_mediator


