Root-Cause Analysis and Explanation
===================================

DoWhy's graphical causal model framework offers powerful tools for root cause analysis and explanation of observed
effects by making use of the decomposition of nodes in a causal graph into single causal mechanisms.

**Attribute Anomalies:** Given anomalous observations (i.e., outliers), we can attribute them to the nodes that have
caused them. The idea here is to ask the counterfactual question "If node x had behaved differently, would we
still have observed the anomaly?". The change in our target node quantifies how much a node contributes to the
observed anomaly. To ensure a fair attribution, this needs to be estimated systematically for different combinations
of changes in nodes.

**Distribution Change Attribution:** Given two data sets where the distribution has changed, we attribute the changes to
the nodes in the graph that have caused the changes. Here, we first identify the data generating mechanisms of nodes that
have changed and then attribute changes in distribution (e.g., with respect to a change in the variance or in KL divergence)
to these mechanisms.

**Feature Relevance:** Here, we address the question of how relevant a feature is for a model. Popular packages like
SHAP address this question by defining a certain set function and estimating the Shapley value for each input feature.
In our case, we do something similar, but also incorporate the probabilistic nature of a causal mechanism. That is,
we also incorporate the noise that influences a variable. By this, we can compute the relevance of inputs, but can also
incorporate the unexplainable part represented by the noise. Furthermore, the Shapley value estimator in DoWhy offers a
flexible way to define a customized set function.

.. toctree::
   :maxdepth: 1

   anomaly_attribution
   distribution_change
   feature_relevance
