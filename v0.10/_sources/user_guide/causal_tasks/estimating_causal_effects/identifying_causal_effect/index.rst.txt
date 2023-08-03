Identifying causal effect
=========================

Given a causal graph and the set of observed variables, identification of causal effect  is the process of determining whether the effect can be estimated using the available variables' data. Formally, identification takes the target causal effect expression, e.g., :math:`E[Y|do(A)]`, and converts it to a form that can be estimated using observed data distribution, i.e., without the do-operator.

For an introduction to identification in causal inference, check out the `book chapter <https://causalinference.gitlab.io/causal-reasoning-book-chapter3/>`_.


.. toctree::
   :maxdepth: 1

   backdoor
   frontdoor
   instrumental_variable
   id_algorithm
