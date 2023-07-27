Estimating average causal effect using backdoor
===============================================

Effect estimation with backdoor amounts to estimating a conditional probability distribution. Given an action A, an outcome Y and set of backdoor variables W, the causal effect is identified as, :math:`\sum_wE[Y|A,W=w]P(W=w)`.

We can use any estimator that can output conditional expectation. DoWhy supports the following three types of estimators for average causal effect. 


.. toctree::
   :maxdepth: 1

   regression_based_methods
   distance_matching
   propensity_based_methods
   do_sampler

