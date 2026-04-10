Graph Refutation
================

Graph refutation tests whether the conditional independence constraints implied by the causal graph are
consistent with the observed data. It leverages the Local Markov Conditions (LMCs) — each node in the graph
should be conditionally independent of its non-descendants given its parents.

If the dataset violates any of these conditional independence constraints, then the assumed causal graph
may be incorrect and downstream causal estimates may be unreliable.

The graph refuter uses partial correlation for continuous data and conditional mutual information for
discrete data to test each implied conditional independence statement.

.. code-block:: python

    from dowhy import CausalModel

    model = CausalModel(
        data=data,
        treatment="treatment",
        outcome="outcome",
        graph=graph
    )
    res = model.refute_graph(k=1, independence_test={"test_for_continuous": "partial_correlation",
                                                      "test_for_discrete": "conditional_mutual_information"})
    print(res)

For a detailed walkthrough, see the `Graph Conditional Independence Refuter notebook <../../../example_notebooks/graph_conditional_independence_refuter.html>`_.
