Specifying a causal graph using domain knowledge
================================================

In DoWhy, we use the `NetworkX <https://networkx
.github.io/>`__ library to create causal graphs. In the snippet below, we create a chain
X→Y→Z:

>>> import networkx as nx
>>> causal_graph = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])

The networkx graph object can then be passed directly to DoWhy functions. Alternatively, you can instantiate a `CausalModel` using the GML string representation of the networkx graph.

>>> from dowhy import CausalModel
>>> import networkx as nx
>>> model = CausalModel(
>>>    data=df, # some pandas dataframe
>>>    treatment="v0",
>>>    outcome="y",
>>>    graph="\n".join(nx.generate_gml(causal_graph))
>>> )

Note that, depending on the causal task, the graph doesn’t need to be completely specified. For instance, for
:doc:`../causal_tasks/estimating_causal_effects/effect_estimation_with_estimators` one can provide a partial graph,
representing prior knowledge
about some of the variables. DoWhy automatically considers the rest of the variables as potential confounders. Or, alternatively, one can provide the names of the variables needed for identification of the target causal quantity, e.g., instrumental variables or common causes of treatment and outcome for the effect estimation task. 

>>> model = CausalModel(
>>>    data=df, # some pandas dataframe
>>>    treatment="v0",
>>>    outcome="y",
>>>    common_causes=["w", "z"],
>>> )
