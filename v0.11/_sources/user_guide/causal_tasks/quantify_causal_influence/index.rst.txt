Quantify Causal Influence
=========================

In addition to estimating the average *total* causal effect, DoWhy can also be used for mediation analysis, for
estimating the direct arrow strength between two nodes and for estimating the intrinsic causal influence of nodes in
a causal graph.

The main differences between these methods are:

**Mediation Analysis:** Mediation analysis primarily aims to decompose the total effect of a treatment on an outcome
into direct and indirect effects through one or more mediator variables. The focus is on understanding how much of the
effect is transmitted directly from the treatment to the outcome and how much is mediated through other variables.

**Direct Arrow Strength:** This method quantifies the causal influence of one variable on another by measuring the
change in the distribution when an edge in a graph is removed. It uses a specific measure for estimating the change,
such as the difference in variance or the relative entropy when removing an edge. This provides an understanding of how the
removal of a specific causal link affects the target variable and, by this, provides a single value representing the
strength of a specific causal link that is well defined for nonlinear relationships.

**Intrinsic Causal Contribution:** This method focuses on estimating the intrinsic contribution of a node within a
graph, independent of the influences inherited from its ancestors. It involves representing each node as a function of
upstream noise terms and employs structure-preserving interventions to measure the influence of the noise terms of each
node. By this, this method separates the inherent information added by each node from that obtained from its ancestors.
The information added can be quantified by measures like (conditional) variance or entropy etc.

.. toctree::
   :maxdepth: 1

   mediation_analysis
   quantify_arrow_strength
   icc
