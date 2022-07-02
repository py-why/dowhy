import networkx as nx
import numpy as np

EXCEPTION_COND = "Conditions to guarantee the existence of an optimal adjustment set are not satisfied"
EXCEPTION_NO_ADJ = "An adjustment set formed by observable variables does not exist"

class ConditionException(Exception):
    pass


class NoAdjException(Exception):
    pass


class CausalGraph(nx.DiGraph):
    """
    A class for Causal Graphs. Inherits from nx.Digraph.

    Implements methods for finding optimal adjustment sets.
    """

    def __init__(self):
        super().__init__(self)

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
            ancestors_node = nx.ancestors(self, node)
            ancestors = ancestors.union(ancestors_node)

        ancestors = ancestors.union(set(nodes))

        return ancestors

    def backdoor_graph(self, treatment, outcome):
        """Returns the back-door graph associated with treatment and outcome

        Parameters
        ----------
        treatment : string
           A node in the graph
        outcome : string
           A node in the graph

        Returns
        ----------
        Gbd: nx.DiGraph()
        """
        Gbd = self.copy()

        for path in nx.all_simple_edge_paths(self, treatment, outcome):
            first_edge = path[0]
            Gbd.remove_edge(first_edge[0], first_edge[1])

        return Gbd

    def causal_vertices(self, treatment, outcome):
        """Returns the set of all vertices that lie in a causal path between treatment and outcome.

        Parameters
        ----------
        treatment : string
           A node in the graph
        outcome : string
           A node in the graph

        Returns
        ----------
        causal_vertices: set
        """
        causal_vertices = set()
        causal_paths = list(nx.all_simple_paths(self, source=treatment, target=outcome))

        for path in causal_paths:
            causal_vertices = causal_vertices.union(set(path))

        causal_vertices.remove(treatment)

        return causal_vertices

    def forbidden(self, treatment, outcome):
        """Returns the forbidden set with respect to treatment and outcome

        Parameters
        ----------
        treatment : string
           A node in the graph
        outcome : string
           A node in the graph

        Returns
        ----------
        forbidden: set
        """
        forbidden = set()

        for node in self.causal_vertices(treatment, outcome):
            forbidden = forbidden.union(nx.descendants(self, node).union(node))

        return forbidden.union(treatment)

    def ignore(self, treatment, outcome, L, N):
        """Returns the set of ignorable vertices with respect to treatment, outcome,
        L and N. Used in the construction of the H0 and H1 graphs.

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        ignore: set
        """
        set1 = set(self.ancestors_all(L + [treatment, outcome]))
        set1.remove(treatment)
        set1.remove(outcome)

        set2 = set(self.nodes()) - set(N)
        set2 = set2.union(self.forbidden(treatment, outcome))

        ignore = set1.intersection(set2)

        return ignore

    @staticmethod
    def unblocked(H, treatment, Z):
        """Returns the unblocked set of Z with respect to treatment

        Parameters
        ----------
        H : nx.Graph()
            Undirected graph
        treatment : string
            A node in the graph
        Z : list of strings
            Nodes in the graph

        Returns
        ----------
        unblocked: set
        """

        G2 = H.subgraph(H.nodes() - set(Z))

        B = nx.node_connected_component(G2, treatment)

        unblocked = set(nx.node_boundary(H, B))
        return unblocked

    def build_H0(self, treatment, outcome, L):
        """Returns the H0 graph associated with treatment, outcome and L

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph

        Returns
        ----------
        H0: nx.Graph()
        """
        # restriction to ancestors
        anc = self.ancestors_all(L + [treatment, outcome])
        G2 = self.subgraph(anc)

        # back-door graph
        G3 = G2.backdoor_graph(treatment, outcome)

        # moralization
        H0 = nx.moral_graph(G3)

        return H0

    def build_H1(self, treatment, outcome, L, N):
        """Returns the H0 graph associated with treatment, outcome, L and N

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        H1: nx.Graph()
        """
        H0 = self.build_H0(treatment, outcome, L)

        ignore_nodes = self.ignore(treatment, outcome, L, N)

        H1 = H0.copy().subgraph(H0.nodes() - ignore_nodes)
        H1 = nx.Graph(H1)
        vertices_list = list(H1.nodes())

        for i, node1 in enumerate(vertices_list):
            for node2 in vertices_list[(i + 1) :]:
                for path in nx.all_simple_paths(H0, source=node1, target=node2):
                    if set(path).issubset(ignore_nodes.union(set([node1, node2]))):
                        H1.add_edge(node1, node2)
                        break
        for node in L:
            H1.add_edge(treatment, node)
            H1.add_edge(node, outcome)

        return H1

    def build_D(self, treatment, outcome, L, N):
        """Returns the D flow network associated with treatment, outcome, L and N.
        If a node does not have a 'cost' attribute, this function will assume
        the cost is infinity

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        D: nx.DiGraph()
        """
        H1 = self.build_H1(treatment, outcome, L, N)
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

    def compute_smallest_mincut(self, treatment, outcome, L, N):
        """Returns a min-cut in the flow network D associated with
        treatment, outcome, L and N that is contained in any other min-cut

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        S_c: set
        """
        D = self.build_D(treatment=treatment, outcome=outcome, L=L, N=N)
        _, flow_dict = nx.algorithms.flow.maximum_flow(
            flowG=D, _s=outcome + "''", _t=treatment + "'"
        )
        queu = [outcome + "''"]
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
        for node in self.nodes:
            nodep = node + "'"
            nodepp = node + "''"
            condition = nodep in S and nodepp not in S
            if condition:
                Z.add(node)
        return Z

    def optimal_adj_set(self, treatment, outcome, L, N):
        """Returns the optimal adjustment set with respect to treatment, outcome, L and N

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        optimal: set
        """
        H1 = self.build_H1(treatment, outcome, L, N)
        if treatment in H1.neighbors(outcome):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        elif N == self.nodes() or set(N).issubset(
            self.ancestors_all(L + [treatment, outcome])
        ):
            optimal = nx.node_boundary(H1, set([outcome]))
            return optimal
        else:
            raise ConditionException(EXCEPTION_COND)

    def optimal_minimal_adj_set(self, treatment, outcome, L, N):
        """Returns the optimal minimal adjustment set with respect to treatment, outcome, L and N

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        optimal_minimal: set
        """

        H1 = self.build_H1(treatment, outcome, L, N)

        if treatment in H1.neighbors(outcome):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        else:
            optimal_minimal = self.unblocked(
                H1, treatment, nx.node_boundary(H1, set([outcome]))
            )
            return optimal_minimal

    @staticmethod
    def isInMinimum(H, treatment, outcome, node):
        """Returns true if and only if node is a member of a minimum size vertex
        cut between treatment and outcome in H

        Parameters
        ----------
        H : nx.Graph()
            Undirected graph
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        node : string
            A node in the graph

        Returns
        ----------
        is_in_minimum: bool
        """
        m1 = len(nx.minimum_node_cut(H, treatment, outcome))

        H_mod = H.copy()
        H_mod.add_edge(treatment, node)
        H_mod.add_edge(outcome, node)

        m2 = len(nx.minimum_node_cut(H_mod, treatment, outcome))

        is_in_minimum = m1 == m2

        return is_in_minimum

    def optimal_minimum_adj_set(self, treatment, outcome, L, N):
        """Returns the optimal minimum adjustment set with respect to treatment, outcome, L and N

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        optimal_minimum: set
        """

        H1 = self.build_H1(treatment, outcome, L, N)

        optimal_minimum = set()

        if treatment in H1.neighbors(outcome):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        else:
            if outcome not in nx.node_connected_component(H1, treatment):
                return optimal_minimum

            for path in nx.node_disjoint_paths(H1, s=outcome, t=treatment):
                for node in path:
                    if node == outcome:
                        continue
                    if self.isInMinimum(H1, treatment, outcome, node):
                        optimal_minimum.add(node)
                        break
            return optimal_minimum

    def optimal_mincost_adj_set(self, treatment, outcome, L, N):
        """Returns the optimal minimum cost adjustment set with respect to treatment, outcome, L and N

        Parameters
        ----------
        treatment : string
            A node in the graph
        outcome : string
            A node in the graph
        L : list of strings
            Nodes in the graph
        N : list of strings
            Nodes in the graph

        Returns
        ----------
        optimal_mincost: set
        """
        H1 = self.build_H1(treatment, outcome, L, N)
        if treatment in H1.neighbors(outcome):
            raise NoAdjException(EXCEPTION_NO_ADJ)
        else:
            S_c = self.compute_smallest_mincut(
                treatment=treatment, outcome=outcome, L=L, N=N
            )
            optimal_mincost = self.h_operator(S_c)
        return optimal_mincost
