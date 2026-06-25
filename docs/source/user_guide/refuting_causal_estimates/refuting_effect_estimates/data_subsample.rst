Data Subsample Refuter
======================

The Data Subsample refuter asks: Does the estimated effect change significantly when we replace the given
dataset with a randomly selected subset?

**Type**: Invariant transformation (negative control)

**Null hypothesis (H₀)**: The true causal effect equals the new estimated effect on the subsample.
Since subsampling does not change the underlying causal mechanism, the estimate should remain stable
(within sampling error) across subsamples.

**Expected outcome**: A reliable causal estimate should not depend strongly on which particular
observations are included. The p-value measures how often, across subsampled datasets, the estimate
would differ from the original by as much or more than observed — by chance.

Interpreting the p-value
-------------------------

* **p-value ≥ 0.05** ✅: The estimate is stable across data subsamples. This supports the robustness
  of the causal estimate to the specific dataset used.

* **p-value < 0.05** ⚠️: The estimate changes significantly when using a subsample. This may indicate:
  (a) the effect estimate is unstable due to small sample size, (b) the effect estimate is driven by
  a small group of high-leverage observations, or (c) the model is over-fitting to the full dataset.

What to do if the test fails
-----------------------------

1. Try a smaller ``subset_fraction`` (e.g. 0.6) to test stability more aggressively, or a larger
   one (e.g. 0.9) if your dataset is small.
2. Check for high-leverage observations that disproportionately affect the estimate.
3. Increase ``num_simulations`` to stabilise the p-value.
4. Consider whether the sample is large enough to reliably estimate the effect.

.. code-block:: python

    res_subset = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        show_progress_bar=True,
        subset_fraction=0.9,
        num_simulations=100,
    )
    print(res_subset)
