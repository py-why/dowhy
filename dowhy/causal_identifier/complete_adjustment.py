import networkx as nx

import pywhy_graphs

class CompleteAdjustment:

    def __init__(self, graph, x, y, z=None):
        self._graph = graph
        self._X = x
        self._Y = y
        if z is None:
            self._Z = set()
        else:
            self._Z = z

    def adjustable(self, G):
        #check amenability
        if not self._is_amenable():
            return False
        
        #check if z contains any node from the forbidden set

        if not self._check_forbidden_set():
            return False

        #find the proper back-door graph
        proper_back_door_graph = self._proper_backdoor_graph()

        #check if z m-seperates x and y in Gpbd
        if not pywhy_graphs.m_seperated(proper_back_door_graph, self._X, self._Y, self._Z):
            return False
        
        return True

    def _is_amenable(self):
        dp = self._graph.directed_paths(self._graph, self._X, self._Y)
        pdp = pywhy_graphs.possibly_directed_paths(self._graph, dp)
        ppdp = pywhy_graphs.proper_paths(self._graph, pdp)
        visible_edges = frozenset(pywhy_graphs.get_visible_edges(self._graph, self._X))
        for elem in ppdp:
            first_edge = elem[0]
            if first_edge in visible_edges and first_edge[0] in self._X:
                continue
            else:
                return False
        return True
    
    def _check_forbidden_set(self):
        forbidden_set = pywhy_graphs.find_forbidden_set(self._graph, self._X, self._Y)
        if len(self._Z.intersection(forbidden_set)) > 0:
            return False
        else:
            return True

    def _proper_backdoor_graph(self):
        dp = self._graph.directed_paths(self._X, self._Y)
        pdp = pywhy_graphs.possibly_directed_paths(self._graph, dp)
        ppdp = pywhy_graphs.proper_paths(self._graph, pdp)
        visible_edges = pywhy_graphs.get_visible_edges(self._graph) # assuming all are directed edges
        x_vedges = []
        for elem in visible_edges:
            if elem[0] in self._X:
                x_vedges.append(elem)
        x_vedges = frozenset(x_vedges)
        all_edges = []
        for elem in ppdp:
            all_edges.extend(elem)
        all_edges = frozenset(all_edges)
        to_remove = all_edges.intersection(x_vedges)
        G = self._graph.copy()
        for elem in to_remove:
            G.remove_edge(elem[0], elem[1], G.directed_edge_name)
        return G