import pytest

from pywhy_graphs import MAG, PAG

from dowhy.causal_identifier.complete_adjustment import CompleteAdjustment

def test_complete_adjsutment():
    G = MAG()
    G.add_edge("Z", "X", G.directed_edge_name)
    G.add_edge("Z", "Y", G.directed_edge_name)
    G.add_edge("X", "Y", G.directed_edge_name)

    cad = CompleteAdjustment()

    assert cad.adjustable(G)

    G = MAG()
    G.add_edge("X", "Z", G.directed_edge_name)
    G.add_edge("X", "Y", G.directed_edge_name)

    cad = CompleteAdjustment()

    assert cad.adjustable(G)

    G = MAG()
    G.add_edge("X", "Z", G.directed_edge_name)
    G.add_edge("Z", "Y", G.directed_edge_name)
    G.add_edge("U", "X", G.directed_edge_name)
    G.add_edge("U", "Y", G.directed_edge_name)

    cad = CompleteAdjustment()

    assert not cad.adjustable(G)

    G = PAG()
    G.add_edge("I", "X", G.directed_edge_name)
    G.add_edge("Z", "X", G.directed_edge_name)
    G.add_edge("A", "X", G.directed_edge_name)
    G.add_edge("X", "Y", G.directed_edge_name)
    G.add_edge("Z", "Y", G.directed_edge_name)
    G.add_edge("B", "Y", G.directed_edge_name)
    G.add_edge("B", "Z", G.circle_edge_name)
    G.add_edge("Z", "B", G.circle_edge_name)
    G.add_edge("A", "B", G.circle_edge_name)
    G.add_edge("B", "A", G.circle_edge_name)
    G.add_edge("A", "Z", G.circle_edge_name)
    G.add_edge("Z", "A", G.circle_edge_name)
    G.add_edge("A", "I", G.circle_edge_name)
    G.add_edge("I", "A", G.circle_edge_name)

    cad = CompleteAdjustment()

    assert cad.adjustable(G)

