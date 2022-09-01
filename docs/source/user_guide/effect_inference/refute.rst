Refute the obtained estimate
-------------------------------------
Having access to multiple refutation methods to validate an effect estimate from a
causal estimator is
a key benefit of using DoWhy.

Supported refutation methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Add Random Common Cause**: Does the estimation method change its estimate after
  we add an independent random variable as a common cause to the dataset?
  (*Hint: It should not*)
* **Placebo Treatment**: What happens to the estimated causal effect when we
  replace the true treatment variable with an independent random variable?
  (*Hint: the effect should go to zero*)
* **Dummy Outcome**: What happens to the estimated causal effect when we replace
  the true outcome variable with an independent random variable? (*Hint: The
  effect should go to zero*)
* **Simulated Outcome**: What happens to the estimated causal effect when we
  replace the dataset with a simulated dataset based on a known data-generating
  process closest to the given dataset? (*Hint: It should match the effect parameter
  from the data-generating process*)
* **Add Unobserved Common Causes**: How sensitive is the effect estimate when we
  add an additional common cause (confounder) to the dataset that is correlated
  with the treatment and the outcome? (*Hint: It should not be too sensitive*)
* **Data Subsets Validation**: Does the estimated effect change significantly when
  we replace the given dataset with a randomly selected subset? (*Hint: It
  should not*)
* **Bootstrap Validation**: Does the estimated effect change significantly when we
  replace the given dataset with bootstrapped samples from the same dataset? (*Hint: It should not*)

Examples of using refutation methods are in the `Refutations <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_refuter_notebook.ipynb>`_ notebook. For an advanced refutation that uses a simulated dataset based on user-provided or learnt data-generating processes, check out the `Dummy Outcome Refuter <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_demo_dummy_outcome_refuter.ipynb>`_ notebook.
As a practical example, `this notebook <https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_refutation_testing.ipynb>`_ shows an application of refutation methods on evaluating effect estimators for the Infant Health and Development Program (IHDP) and Lalonde datasets.

