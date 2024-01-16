from typing import List

import networkx as nx
import numpy as np

EXCEPTION_NO_ADJ = "An adjustment set formed by observable variables does not exist"
EXCEPTION_COND_NO_OPT = "Conditions to guarantee the existence of an optimal adjustment set are not satisfied"


class EfficientBackdoor:
    """
    Implements methods for finding optimal (efficient) backdoor sets.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        action_nodes: List[str],
        outcome_nodes: List[str],
        observed_nodes: List[str],
        conditional_node_names=None,
        costs=None,
    ):
        """
        :param graph: nx.DiGraph
            A causal graph.
        :param costs: list
            A list with non-negative costs associated with variables in the graph. Only used
            for estimatand_type='non-parametric-ate' and method_name='efficient-mincost-adjustment'. If
            not costs are provided by the user, and method_name='efficient-mincost-adjustment', costs
            are assumed to be equal to one for all variables in the graph. The structure of the list should
            be of the form [(node, {"cost": x}) for node in nodes].
        :param conditional_node_names: list
            A list with variables that are used to determine treatment. If none are
            provided, it is assumed that the intervention sets the treatment to a constant.
        """
        assert (
            len(action_nodes) == 1
        ), "The methods for computing efficient backdoor sets are only valid for one dimensional treatments"
        assert (
            len(outcome_nodes) == 1
        ), "The methods for computing efficient backdoor sets are only valid for one dimensional outcomes"
        self.graph = graph
        if costs is None:
            # If no costs are passed, use uniform costs
            costs = [(node, {"cost": 1}) for node in self.graph.nodes]
        assert all([tup["cost"] > 0 for _, tup in costs]), "All costs must be positive"
        self.graph.add_nodes_from(costs)
        self.observed_nodes = set([node for node in self.graph.nodes if node in set(observed_nodes)])
        if conditional_node_names is None:
            conditional_node_names = []
        assert set(conditional_node_names).issubset(
            self.observed_nodes
        ), "Some conditional variables are not marked as observed"
        self.conditional_node_names = conditional_node_names

        self.treatment_name = action_nodes[0]
        self.outcome_name = outcome_nodes[0]

    def ancestors_all(self, nodes):
        """Method to compute the set of all ancestors of a set of nodes.
        A node is always an ancestor of itself.


        :param nodes: list
            A list of nodes in the graph.
        :returns ancestors: set
            The set of nodes that are ancestors of nodes in nodes.
        """
        ancestors = set()

        for node in nodes:
            ancestors_node = nx.ancestors(self.graph, node)
            ancestors = ancestors.union(ancestors_node)

        ancestors = ancestors.union(set(nodes))

        return ancestors

    def backdoor_graph(self, G):
        """Method to compute the proper back-door graph associated with
         treatment and outcome.

        :param G: nx.DiGraph
            A directed acyclic graph.
        :returns Gbd: nx.DiGraph
            The proper backdoor graph of G.
        """
        Gbd = G.copy()

        for path in nx.all_simple_edge_paths(G, self.treatment_name, self.outcome_name):
            first_edge = path[0]
            Gbd.remove_edge(first_edge[0], first_edge[1])

        return Gbd

    def causal_vertices(self):
        """Method to compute the set of all vertices that lie in a causal path
         between treatment and outcome.

        :returns causal_vertices: set
            A set with vertices lying on some causal path between treatment and
            outcome.
        """
        causal_vertices = set()
        causal_paths = list(
            nx.all_simple_paths(
                self.graph,
                source=self.treatment_name,
                target=self.outcome_name,
            )
        )

        for path in causal_paths:
            causal_vertices = causal_vertices.union(set(path))

        causal_vertices.remove(self.treatment_name)

        return causal_vertices

    def forbidden(self):
        """Method to compute the forbidden set with respect to treatment and
         outcome.

        :returns forbidden: set
            The forbidden set.
        """
        forbidden = set()

        for node in self.causal_vertices():
            forbidden = forbidden.union(nx.descendants(self.graph, node).union({node}))

        return forbidden.union({self.treatment_name})

    def ignore(self):
        """Method to compute the set of ignorable vertices with respect to
         treatment, outcome, conditional and observable variables.
         Used in the construction of the H0 and H1 graphs. See Smucler,
         Sapienza and Rotnitzky (2021), Biometrika, for the full definition
         of this set.

        :returns ignore: set
            The set of ignorable vertices.
        """
        set1 = set(self.ancestors_all(self.conditional_node_names + [self.treatment_name, self.outcome_name]))
        set1.remove(self.treatment_name)
        set1.remove(self.outcome_name)

        set2 = set(self.graph.nodes()) - self.observed_nodes
        set2 = set2.union(self.forbidden())

        ignore = set1.intersection(set2)

        return ignore

    def unblocked(self, H, Z):
        """Method to compute the unblocked set of Z with respect to treatment.
        See Smucler, Sapienza and Rotnitzky (2021), Biometrika, for the full
        definition of this set.

        :params H: nx.Graph
            An undirected graph.
        :param Z: list
            A list with nodes in the graph H.
        :returns unblocked: set
            The unblocked set.
        """

        G2 = H.subgraph(H.nodes() - set(Z))

        B = nx.node_connected_component(G2, self.treatment_name)

        unblocked = set(nx.node_boundary(H, B))
        return unblocked

    def build_H0(self):
        """Returns the H0 graph associated with treatment, outcome, conditional
        and observable variables. See Smucler, Sapienza and Rotnitzky (2021),
        Biometrika, for the full definition of this graph.

        :returns H0: nx.Graph
            The H0 graph.
        """
        # restriction to ancestors
        anc = self.ancestors_all(self.conditional_node_names + [self.treatment_name, self.outcome_name])
        G2 = self.graph.subgraph(anc)

        # back-door graph
        G3 = self.backdoor_graph(G2)

        # moralization
        H0 = nx.moral_graph(G3)

        return H0

    def build_H1(self):
        """Returns the H1 graph associated with treatment, outcome, conditional
        and observable variables. See Smucler, Sapienza and Rotnitzky (2021),
        Biometrika, for the full definition of this graph.

        :returns H1: nx.Graph
            The H1 graph.
        """
        H0 = self.build_H0()

        ignore_nodes = self.ignore()

        H1 = H0.copy().subgraph(H0.nodes() - ignore_nodes)
        H1 = nx.Graph(H1)
        vertices_list = list(H1.nodes())

        for i, node1 in enumerate(vertices_list):
            for node2 in vertices_list[(i + 1) :]:
                for path in nx.all_simple_paths(H0, source=node1, target=node2):
                    if set(path).issubset(ignore_nodes.union({node1, node2})):
                        H1.add_edge(node1, node2)
                        break

        for node in self.conditional_node_names:
            H1.add_edge(self.treatment_name, node)
            H1.add_edge(node, self.outcome_name)

        return H1

    def build_D(self):
        """Returns the D flow network associated with treatment, outcome,
        conditional and observable variables. If a node does not have a 'cost'
        attribute, this function will assume the cost is infinity.

        See Smucler and Rotnitzky (2022), Journal of Causa Inference, for the
        full definition of this flow network.

        :returns D: nx.DiGraph
            The D flow network.
        """
        H1 = self.build_H1()
        D = nx.DiGraph()
        for node in H1.nodes.keys():
            if "cost" in H1.nodes[node]:
                capacity = H1.nodes[node]["cost"]
            else:
                capacity = np.inf
            D.add_edge(node + "'", node + "''", capacity=capacity)

        for edge in H1.edges.keys():
            D.add_edge(edge[0] + "''", edge[1] + "'", capacity=np.inf)
            D.add_edge(edge[1] + "''", edge[0] + "'", capacity=np.inf)
        return D

    def compute_smallest_mincut(self):
        """Returns a min-cut in the flow network D associated with
        treatment, outcome, conditional and observable variables that is
        contained in any other min-cut.

        :returns S_c: set
            The min-cut with the above property.
        """
        D = self.build_D()
        _, flow_dict = nx.algorithms.flow.maximum_flow(
            flowG=D,
            _s=self.outcome_name + "''",
            _t=self.treatment_name + "'",
        )
        queu = [self.outcome_name + "''"]
        S_c = set()
        visited = set()
        while len(queu) > 0:
            node = queu.pop()
            if node not in visited:
                visited.add(node)
                point_in = D.in_edges(node)
                point_out = D.out_edges(node)
                for edge in point_out:
                    capacity = D.edges[edge]["capacity"]
                    flow = flow_dict[edge[0]][edge[1]]
                    if flow < capacity:
                        queu.append(edge[1])
                        S_c.add(edge[1])
                for edge in point_in:
                    flow = flow_dict[edge[0]][edge[1]]
                    if flow > 0:
                        queu.append(edge[0])
                        S_c.add(edge[0])
        return S_c

    def h_operator(self, S):
        """Given a set S of vertices in the flow network D, returns the
        operator h(S), a set of vertices in the undirected graph H1.

        See Smucler and Rotnitzky (2022), Journal of Causal Inference, for the
        full definition of this operator.

        :param S: set
            A set of vertices in the flow network D associated treatment,
            outcome, conditional and observable variables.
        :returns Z: set
            The set obtained from applying the h operator to S.
        """
        Z = set()
        for node in self.graph.nodes:
            nodep = node + "'"
            nodepp = node + "''"
            condition = nodep in S and nodepp not in S
            if condition:
                Z.add(node)
        return Z

    def optimal_adj_set(self):
        """Returns the optimal adjustment set with respect to treatment,
        outcome, conditional and observable variables.

        If the sufficient conditions for the existence of the optimal adjustment
        set outlined in Smucler, Sapienza and Rotnitzky (2021), Biometrika, do
        not hold, an error is raised.

        :returns: optimal: set
            The optimal adjustment set.
        """
        H1 = self.build_H1()
        if self.treatment_name in H1.neighbors(self.outcome_name):
            raise ValueError(EXCEPTION_NO_ADJ)
        elif self.observed_nodes == set(self.graph.nodes()) or self.observed_nodes.issubset(
            self.ancestors_all(self.conditional_node_names + [self.treatment_name, self.outcome_name])
        ):
            optimal = nx.node_boundary(H1, {self.outcome_name})
            return optimal
        else:
            raise ValueError(EXCEPTION_COND_NO_OPT)

    def optimal_minimal_adj_set(self):
        """Returns the optimal minimal adjustment set with respect to treatment,
        outcome, conditional and observable variables.

        :returns: optimal_minimal: set
            The optimal minimal adjustment set.
        """

        H1 = self.build_H1()

        if self.treatment_name in H1.neighbors(self.outcome_name):
            raise ValueError(EXCEPTION_NO_ADJ)
        else:
            optimal_minimal = self.unblocked(H1, nx.node_boundary(H1, [self.outcome_name]))
            return optimal_minimal

    def optimal_mincost_adj_set(self):
        """Returns the optimal minimum cost adjustment set with respect to
        treatment, outcome, conditional and observable variables.

        Note that when the costs are constant, this is the optimal adjustment
        set among those of minimum cardinality.

        :returns: optimal_mincost: set
            The optimal minimum cost adjustment set.
        """
        H1 = self.build_H1()
        if self.treatment_name in H1.neighbors(self.outcome_name):
            raise ValueError(EXCEPTION_NO_ADJ)
        else:
            S_c = self.compute_smallest_mincut()
            optimal_mincost = self.h_operator(S_c)
        return optimal_mincost
