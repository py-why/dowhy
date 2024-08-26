import networkx as nx

import pywhy_graphs

def adjustable(self, G, X, Y, Z=None):

    if Z is None:
        Z = set()

    #check amenability
    if not self._is_amenable():
        return False
    
    #check if z contains any node from the forbidden set

    if not self._check_forbidden_set():
        return False

    #find the proper back-door graph
    proper_back_door_graph = self._proper_backdoor_graph()

    #check if z m-seperates x and y in Gpbd
    if not pywhy_graphs.m_seperated(proper_back_door_graph, X, Y, Z):
        return False
    
    return True

def _is_amenable(G, X, Y):
    dp = G.directed_paths(G, X, Y)
    pdp = pywhy_graphs.possibly_directed_paths(G, dp)
    ppdp = pywhy_graphs.proper_paths(G, pdp)
    visible_edges = frozenset(pywhy_graphs.get_visible_edges(G, X))
    for elem in ppdp:
        first_edge = elem[0]
        if first_edge in visible_edges and first_edge[0] in X:
            continue
        else:
            return False
    return True

def _check_forbidden_set(G,X,Y,Z):
            
    if Z is None:
        Z = set()

    forbidden_set = pywhy_graphs.find_forbidden_set(G, X, Y)
    if len(Z.intersection(forbidden_set)) > 0:
        return False
    else:
        return True

def _proper_backdoor_graph(G,X,Y):
    ppdp = pywhy_graphs.proper_possibly_directed_path(G, X, Y)
    visible_edges = pywhy_graphs.get_visible_edges(G) # assuming all are directed edges
    x_vedges = []
    for elem in visible_edges:
        if elem[0] in X:
            x_vedges.append(elem)
    x_vedges = frozenset(x_vedges)
    all_edges = []
    for elem in ppdp:
        all_edges.extend(elem)
    all_edges = frozenset(all_edges)
    to_remove = all_edges.intersection(x_vedges)
    G = G.copy()
    for elem in to_remove:
        G.remove_edge(elem[0], elem[1], G.directed_edge_name)
    return G