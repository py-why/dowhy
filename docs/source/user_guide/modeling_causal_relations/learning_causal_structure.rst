Learning causal structure from data
===================================

Learning the causal graph is only necessary in case we cannot construct the graph based on domain knowledge or other sources of information. If you already have the causal graph in your specific problem, you can skip this chapter and move on to :doc:`refuting_causal_graph/index`.

In many cases, the true causal structure for a given dataset may not be known. We can utilize *graph discovery* algorithms to provide candidate causal graphs consistent with the dataset. Such graphs can then be inspected, edited and modified by a user to match their domain expertise or world knowledge. Note that these graphs should be not used directly because graph discovery from observational data is a provably impossible problem in the fully non-parametric setting. Given a dataset, there exist multiple graphs that  would lead to the exact same joint distribution and thus are indistinguishable based on the dataset (such graphs constitute the *Markov equivalence class*).  As a result, graph discovery algorithms make certain assumptions to learn a graph and do not guarantee the validity of a learned graph. 

DoWhy does not implement graph discovery algorithms, but provides a simple way to input the learnt graph from a discovery algorithm. The only constraint is that DoWhy expects a directed acyclic graph (DAG) in the form of a networkx graph. For packages that implement different causal discovery algorithms, see the `causal-learn <https://github.com/py-why/causal-learn>`_ and `dodiscover <https://github.com/py-why/dodiscover>`_ packages in PyWhy.

Graph discovery using dodiscover
--------------------------------
TBD

Graph discovery using causal-learn
----------------------------------
TBD
