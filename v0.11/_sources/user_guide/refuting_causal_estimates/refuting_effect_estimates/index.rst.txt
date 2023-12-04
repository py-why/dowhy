Refuting Effect Estimates
=========================

Effect refutations are of two kinds: negative control and sensitivity analysis. 

Refutations based on negative control
-------------------------------------
The first kind of refutation tests are necessary conditions that any good estimation procedure should satisfy. They are also known as *negative controls*. If an estimator fails the refutation test (p-value is <0.05), then it means that there is some problem with the estimator. 

Negative control refutation tests are based on either:

* **Invariant transformations**: Changes in the data that should not change the estimate. Any estimator whose result varies significantly between the original data and the modified data fails the test. Examples are the data subsample and add random common cause refutations. 

* **Nullifying transformations**: After the data change, the causal true estimate is zero. Any estimator whose result varies significantly from zero on the new data fails the test. Examples are the placebo treatment and dummy outcome refutations.

Refutations based on sensitivity analysis
------------------------------------------
The second kind are sensitivity tests that test the robustness of an obtained estimate to violation of assumptions such as no unobserved confounding. 

.. toctree::
   :maxdepth: 2
    
   placebo_treatment
   dummy_outcome
   random_common_cause
   data_subsample
   sensitivity_analysis/index
