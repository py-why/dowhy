Dummy Outcome Refuter
=====================

The Dummy Outcome refuter asks: What happens to the estimated causal effect when we replace the true
outcome variable with an independent random variable (or a simulated outcome from a known
data-generating process)?

**Type**: Nullifying transformation (negative control) for the random-outcome case; ground-truth
validation for the simulated-outcome case.

**Null hypothesis (H₀)**: The true causal effect equals the new estimated effect on the dummy outcome.

**Expected outcome**:

* **Random dummy outcome**: A correct estimator should find no effect (close to zero) when the outcome
  is replaced with a random variable, since there is no causal mechanism linking treatment to noise.
  A low p-value here means the estimator incorrectly finds an effect.
* **Simulated outcome with known effect**: The estimator should recover the known ground-truth effect
  from the specified data-generating process. This is a quantitative validation.

Interpreting the p-value
-------------------------

* **p-value ≥ 0.05** ✅: The estimate is close to zero (or to the known ground-truth) — the estimator
  passes the dummy outcome check.

* **p-value < 0.05** ⚠️: The estimator finds a spurious effect on the dummy outcome. This is a sign of
  model misspecification: the estimator may be capturing a correlation between the treatment and observed
  common causes that leaks into the outcome estimate even when the outcome is pure noise.

What to do if the test fails
-----------------------------

1. Verify that all confounders are correctly specified in the model.
2. Check that the outcome variable is not itself a cause of the treatment (reverse causation).
3. Try a different estimator that is more robust to spurious correlations.

Testing for zero causal effect
-------------------------------

.. code-block:: python

    ref = model.refute_estimate(
        identified_estimand,
        causal_estimate,
        method_name="dummy_outcome_refuter",
    )
    print(ref[0])

Testing for non-zero causal effect (known DGP)
-----------------------------------------------

When you know the true data-generating process, you can validate that the estimator recovers the
known ground-truth effect:

.. code-block:: python

    import numpy as np

    coefficients = np.array([1, 2])
    bias = 3

    def linear_gen(df):
        y_new = np.dot(df[["W0", "W1"]].values, coefficients) + bias
        return y_new

    ref = model.refute_estimate(
        identified_estimand,
        causal_estimate,
        method_name="dummy_outcome_refuter",
        outcome_function=linear_gen,
    )
    print(ref[0])


For a complete example on using the dummy outcome refuter, you can check out the notebook,
:doc:`../../../../example_notebooks/dowhy_demo_dummy_outcome_refuter`.

