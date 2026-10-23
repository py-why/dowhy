Frontdoor criterion
===================

When to use the frontdoor criterion
--------------------------------------

The frontdoor criterion applies when:

* Direct backdoor adjustment is impossible — there is unobserved confounding between
  treatment *X* and outcome *Y*.
* One or more **mediator** variables *M* lie on *every* directed path from *X* to *Y*.
* *M* is not directly affected by the hidden confounder (i.e., all backdoor paths from
  *X* to *M* are blocked), and all backdoor paths from *M* to *Y* are blocked by
  conditioning on *X*.

A classic example is the effect of smoking (X) on cancer (Y) where tar deposits (M)
mediate the effect and the direct X → Y path passes entirely through M.

How it works
------------

The identifying formula (Pearl, 2000) is:

.. math::

   P(Y \mid do(X)) = \sum_M P(M \mid X) \sum_{X'} P(Y \mid M, X') \, P(X')

Intuitively, it chains two backdoor adjustments:

1. Identify the effect of *X* on *M* (backdoor-adjusting with any observed confounders
   of *X* ↔ *M*).
2. Identify the effect of *M* on *Y* (backdoor-adjusting with *X*, which blocks the
   path from the hidden confounder to *Y*).

Identifying with DoWhy
----------------------

Use the same ``identify_effect`` method — DoWhy searches for frontdoor sets
automatically:

>>> import networkx as nx
>>> from dowhy import CausalModel
>>>
>>> # Graph with unobserved confounder (U) and mediator (M)
>>> # X -> M -> Y, U -> X, U -> Y (U is unobserved)
>>> graph = nx.DiGraph([("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")])
>>> model = CausalModel(
...     data=df,
...     treatment="X",
...     outcome="Y",
...     graph=graph,
...     observed_node_names=["X", "M", "Y"],   # U is latent
... )
>>> identified_estimand = model.identify_effect()
>>> print(identified_estimand)

Or equivalently using the functional API:

>>> from dowhy.causal_identifier import identify_effect_auto
>>> from dowhy.graph import build_graph_from_networkx
>>>
>>> causal_graph = build_graph_from_networkx(graph, observed_node_names=["X", "M", "Y"])
>>> identified_estimand = identify_effect_auto(causal_graph, "X", "Y")

When both backdoor and frontdoor sets exist, DoWhy reports all valid strategies.

Pros and cons
-------------

* **Pro**: Identifies causal effects in the presence of unobserved confounders between
  treatment and outcome.
* **Pro**: Requires weaker assumptions than instrumental variables — no exclusion
  restriction on the mediator is needed.
* **Con**: Requires a complete set of mediators on all paths from *X* to *Y*; if any
  direct path bypasses the mediators the criterion fails.
* **Con**: Can be harder to find in practice than backdoor variables.

For an example notebook, see :doc:`../../../../example_notebooks/dowhy_mediation_analysis`.


