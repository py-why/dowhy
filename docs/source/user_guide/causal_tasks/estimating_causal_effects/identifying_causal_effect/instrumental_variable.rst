Natural experiments and instrumental variables
==============================================

When to use instrumental variables
-----------------------------------

The instrumental variable (IV) criterion applies when:

* There is unobserved confounding between treatment *X* and outcome *Y*.
* An **instrument** *Z* is available that:

  1. **Relevance** — *Z* is correlated with (causes) *X*.
  2. **Exclusion restriction** — *Z* affects *Y* only through *X* (no direct path
     from *Z* to *Y*).
  3. **Independence** — *Z* is independent of all hidden confounders of *X* → *Y*.

Classic examples include using distance-to-college as an instrument for years of
education (effect on wages), or a lottery draft number as an instrument for military
service (effect on earnings).

How it works
------------

Under the IV assumptions, with the additional conditions of a binary treatment and
instrument and a monotonicity (no-defiers) assumption, the Local Average Treatment
Effect (LATE / CACE) can be identified via the Wald estimator:

.. math::

   \tau_{LATE} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[X \mid Z=1] - E[X \mid Z=0]}

More generally, two-stage least squares (2SLS) is the standard estimator: regress *X*
on *Z* (first stage), then regress *Y* on the predicted values of *X* (second stage).

Identifying with DoWhy
----------------------

DoWhy detects instruments automatically from the graph:

>>> import networkx as nx
>>> from dowhy import CausalModel
>>>
>>> # Z is an instrument: Z -> X -> Y, with U -> X and U -> Y (unobserved)
>>> graph = nx.DiGraph([("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")])
>>> model = CausalModel(
...     data=df,
...     treatment="X",
...     outcome="Y",
...     graph=graph,
...     observed_node_names=["Z", "X", "Y"],   # U is latent
... )
>>> identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
>>> print(identified_estimand)

Or equivalently using the functional API:

>>> from dowhy.causal_identifier import identify_effect_auto
>>> from dowhy.graph import build_graph_from_networkx
>>>
>>> causal_graph = build_graph_from_networkx(graph, observed_node_names=["Z", "X", "Y"])
>>> identified_estimand = identify_effect_auto(causal_graph, "X", "Y")

Once identified, estimate the effect with the IV estimator:

>>> causal_estimate = model.estimate_effect(
...     identified_estimand,
...     method_name="iv.instrumental_variable",
...     method_params={"iv_instrument_name": "Z"},
... )

Pros and cons
-------------

* **Pro**: Identifies causal effects in the presence of unobserved confounders when a
  valid instrument is available.
* **Con**: The exclusion restriction is *untestable* from data alone — it is a
  modelling assumption that must be justified by domain knowledge.
* **Con**: IV estimates the LATE (effect for compliers), not necessarily the ATE for
  the full population.
* **Con**: Weak instruments (low relevance) lead to inflated variance.

For a full working example, see :doc:`../../../../example_notebooks/dowhy-simple-iv-example`.
