Placebo Treatment Refuter
=========================

The Placebo Treatment refuter asks: What happens to the estimated causal effect when we replace the true
treatment variable with an independent random variable or a permuted version of the original treatment?

**Type**: Nullifying transformation (negative control)

**Null hypothesis (H₀)**: The true causal effect is zero, i.e., the treatment has no effect on the
outcome. The simulated distribution represents what effect estimates look like when treatment is replaced
by noise — this is the null world.

**Expected outcome**: When the treatment is replaced by a random variable, any correct causal estimator
should produce an effect close to zero. The p-value measures how likely it is that the original estimate
would occur under the null hypothesis (no effect).

Interpreting the p-value
-------------------------

* **p-value ≥ 0.05** ✅: The original estimate is significantly different from zero in the placebo
  world — this is the expected result for a real causal effect. The estimator correctly finds no effect
  when treatment is noise.

* **p-value < 0.05** ⚠️: The estimate under the real treatment is not significantly different from
  the placebo estimates. This may indicate that (a) the estimated causal effect is spurious, (b) the
  estimator is not leveraging the true treatment variation, or (c) the sample size is too small to
  reliably detect the effect.

.. note::

   The null hypothesis here is that the **original estimate equals the new effect** (under placebo
   treatment). Because the true causal effect under a placebo should be zero, a non-significant p-value
   means the original estimate is statistically indistinguishable from zero — which is bad news for
   the claimed causal effect.

What to do if the test fails
-----------------------------

1. Check whether the treatment variable has sufficient variation.
2. Ensure the common causes are correctly specified — omitted confounders can make the placebo
   treatment appear to have a real effect.
3. Consider increasing the sample size or using a more efficient estimator.
4. Increase ``num_simulations`` to reduce variability in the p-value.

.. code-block:: python

    res_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        show_progress_bar=True,
        placebo_type="permute",
        num_simulations=100,
    )
    print(res_placebo)

