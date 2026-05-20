Random Common Cause Refuter
===========================

The Add Random Common Cause refuter checks the following: Does the estimation method change its estimate
after we add an independent random variable as a common cause to the dataset?

**Type**: Invariant transformation (negative control)

**Null hypothesis (H₀)**: The true causal effect equals the new estimated effect after adding the random
common cause. Since the added variable is independent of both treatment and outcome, a correct estimator
should not be affected.

**Expected outcome**: If the causal estimator is correctly accounting for the observed confounders, the
estimate should remain stable after adding the random noise variable. The p-value measures how often, across
simulations, the new estimate would differ from the original by as much or more than observed — by chance.

Interpreting the p-value
-------------------------

* **p-value ≥ 0.05** ✅: The estimate is robust. Adding a random confounder does not significantly change
  the estimate — this is the expected behaviour for a well-specified estimator.

* **p-value < 0.05** ⚠️: The estimate changes significantly when a random, unrelated variable is added.
  This may indicate that the estimator is overly sensitive to the set of observed common causes, possibly
  due to model misspecification (e.g., wrong functional form, collinearity, or too few observations).

What to do if the test fails
-----------------------------

1. Check whether the model is over-fitting to the observed common causes.
2. Try a different estimation method (e.g., propensity-score weighting instead of matching).
3. Increase ``num_simulations`` to get a more stable p-value estimate.
4. Examine whether any of the common causes are highly collinear.

.. code-block:: python

    res_random = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        show_progress_bar=True,
        num_simulations=100,
    )
    print(res_random)
