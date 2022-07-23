import networkx as nx
import numpy as np
import warnings

# TODO, how does Dowhy handle this? With exceptions?
EXCEPTION_NO_ADJ = "An adjustment set formed by observable variables does not exist"


class NoAdjException(Exception):
    pass


class EfficientBackdoor:
    """
    Implements methods for finding optimal backdoor sets.
    """

    def __init__(self, graph, conditional_node_names, costs=None):
        self.graph = graph
        self.conditional_node_names = conditional_node_names
        if not costs:
            costs = [(node, {"cost": 1}) for node in self.graph._graph.nodes]
        self.graph._graph.add_nodes_from(costs)
        self.observed_nodes = set(
            [
                node
                for node in self.graph._graph.nodes
                if self.graph._graph.nodes[node]["observed"] == "yes"
            ]
        )

    def ancestors_all(self, nodes):
        """Returns a set with all ancestors of nodes

        Parameters
        ----------
        nodes : list
           A list of nodes in the graph

        Returns
        ----------
        ancestors: set

        Notes
        -----
        A node is always an ancestor of itself.
        """
        ancestors = set()

        for node in nodes:
            ancestors_node = nx.ancestors(self.graph._graph, node)
            ancestors = ancestors.union(ancestors_node)

        ancestors = ancestors.union(set(nodes))

        return ancestors

    def backdoor_graph(self, G):
        """Returns the back-door graph associated with treatment and outcome.

        Returns
        ----------
        Gbd: nx.DiGraph()
        """
        Gbd = G.copy()

        for path in nx.all_simple_edge_paths(
            G, self.graph.treatment_name[0], self.graph.outcome_name[0]
        ):
            first_edge = path[0]
            Gbd.remove_edge(first_edge[0], first_edge[1])

        return Gbd

    def causal_vertices(self):
        """Returns the set of all vertices that lie in a causal path between treatment and outcome.

        Returns
        ----------
        causal_vertices: set
        """
        causal_vertices = set()
        causal_paths = list(
            nx.all_simple_paths(
                self.graph._graph,
                source=self.graph.treatment_name[0],
                target=self.graph.outcome_name[0],
            )
        )

        for path in causal_paths:
            causal_vertices = causal_vertices.union(set(path))

        causal_vertices.remove(self.graph.treatment_name[0])

        return causal_vertices

    def forbidden(self):
        """Returns the forbidden set with respect to treatment and outcome.

        Returns
        ----------
        forbidden: set
        """
        forbidden = set()

        for node in self.causal_vertices():
            forbidden = forbidden.union(
                nx.descendants(self.graph._graph, node).union(node)
            )

        return forbidden.union(self.graph.treatment_name[0])

    def ignore(self):
        """Returns the set of ignorable vertices with respect to treatment, outcome, conditional and observable variables. Used in the construction of the H0 and H1 graphs.

        Returns
        ----------
        ignore: set
        """
        set1 = set(
            self.ancestors_all(
                self.conditional_node_names
                + [self.graph.treatment_name[0], self.graph.outcome_name[0]]
            )
        )
        set1.remove(self.graph.treatment_name[0])
        set1.remove(self.graph.outcome_name[0])

        set2 = set(self.graph._graph.nodes()) - self.observed_nodes
        set2 = set2.union(self.forbidden())

        ignore = set1.intersection(set2)

        return ignore

    def unblocked(self, H, Z):
        """Returns the unblocked set of Z with respect to the treatment variable.

        Parameters
        ----------
        H : nx.Graph()
            Undirected graph
        Z : list of strings
            Nodes in the graph

        Returns
        ----------
        unblocked: set
        """

        G2 = H.subgraph(H.nodes() - set(Z))

        B = nx.node_connected_component(G2, self.graph.treatment_name[0])

        unblocked = set(nx.node_boundary(H, B))
        return unblocked

    def build_H0(self):
        """Returns the H0 graph associated with treatment, outcome, conditional and observable variables.

        Returns
        ----------
        H0: nx.Graph()
        """
        # restriction to ancestors
        anc = self.ancestors_all(
            self.conditional_node_names
            + [self.graph.treatment_name[0], self.graph.outcome_name[0]]
        )
        G2 = self.graph._graph.subgraph(anc)

        # back-door graph
        G3 = self.backdoor_graph(G2)

        # moralization
        H0 = nx.moral_graph(G3)

        return H0

    def build_H1(self):
        """Returns the H1 graph associated with treatment, outcome, conditional and observable variables.

        Returns
        ----------
        H1: nx.Graph()
        """
        H0 = self.build_H0()

        ignore_nodes = self.ignore()

        H1 = H0.copy().subgraph(H0.nodes() - ignore_nodes)
        H1 = nx.Graph(H1)
        vertices_list = list(H1.nodes())

        for i, node1 in enumerate(vertices_list):
            for node2 in vertices_list[(i + 1) :]:
                for path in nx.all_simple_paths(H0, source=node1, target=node2):
                    if set(path).issubset(ignore_nodes.union(set([node1, node2]))):
                        H1.add_edge(node1, node2)
                        break

        for node in self.conditional_node_names:
            H1.add_edge(self.graph.treatment_name[0], node)
            H1.add_edge(node, self.graph.outcome_name[0])

        return H1

    def build_D(self):
        """Returns the D flow network associated with treatment, outcome, conditional and observable variables.
        If a node does not have a 'cost' attribute, this function will assume
        the cost is infinity


        Returns
        ----------
        D: nx.DiGraph()
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
        treatment, outcome, conditional and observable variables that is contained in any other min-cut

        Returns
        ----------
        S_c: set
        """
        D = self.build_D()
        _, flow_dict = nx.algorithms.flow.maximum_flow(
            flowG=D,
            _s=self.graph.outcome_name[0] + "''",
            _t=self.graph.treatment_name[0] + "'",
        )
        queu = [self.graph.outcome_name[0] + "''"]
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
         operator h(S), a set of vertices in the undirected graph H1

        Parameters
        ----------
        S : set
            A set of vertices in D

        Returns
        ----------
        Z: set
        """
        Z = set()
        for node in self.graph._graph.nodes:
            nodep = node + "'"
            nodepp = node + "''"
            condition = nodep in S and nodepp not in S
            if condition:
                Z.add(node)
        return Z

    def optimal_adj_set(self):
        """Returns the optimal adjustment set with respect to treatment, outcome, conditional and observable variables

        Returns
        ----------
        optimal: set
        """
        H1 = self.build_H1()
        if self.graph.treatment_name[0] in H1.neighbors(self.graph.outcome_name[0]):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        elif self.observed_nodes == self.graph._graph.nodes() or self.observed_nodes.issubset(
            self.ancestors_all(
                [self.graph.treatment_name[0], self.graph.outcome_name[0]]
            )
        ):
            optimal = nx.node_boundary(H1, set([self.graph.outcome_name[0]]))
            return optimal
        else:
            warnings.warn(
                "Conditions to guarantee the existence of an optimal adjustment set are not satisfied"
            )
            return None

    def optimal_minimal_adj_set(self):
        """Returns the optimal minimal adjustment set with respect to treatment, outcome, conditional and observable variables

        Returns
        ----------
        optimal_minimal: set
        """

        H1 = self.build_H1()

        if self.graph.treatment_name[0] in H1.neighbors(self.graph.outcome_name[0]):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        else:
            optimal_minimal = self.unblocked(
                H1, nx.node_boundary(H1, set([self.graph.outcome_name[0]]))
            )
            return optimal_minimal

    def optimal_mincost_adj_set(self):
        """Returns the optimal minimum cost adjustment set with respect to treatment, outcome, conditional and observable variables

        Returns
        ----------
        optimal_mincost: set
        """
        H1 = self.build_H1()
        if self.graph.treatment_name[0] in H1.neighbors(self.graph.outcome_name[0]):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        else:
            S_c = self.compute_smallest_mincut()
            optimal_mincost = self.h_operator(S_c)
        return optimal_mincost
