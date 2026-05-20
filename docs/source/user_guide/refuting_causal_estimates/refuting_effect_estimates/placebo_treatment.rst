Placebo Treatment Refuter
=========================

The Placebo Treatment refuter asks: What happens to the estimated causal effect when we replace the true
treatment variable with an independent random variable or a permuted version of the original treatment?

**Type**: Nullifying transformation (negative control)

**Null hypothesis (H₀)**: The true causal effect is zero, i.e., the treatment has no effect on the
outcome. The simulated distribution represents what effect estimates look like when treatment is replaced
by noise — this is the null world.

**Expected outcome**: When the treatment is replaced by a random variable, any correct causal estimator
should produce an effect close to zero. The p-value tests whether the simulated placebo-effect estimates
are consistent with zero, i.e., whether the placebo distribution is centered around the null effect.

Interpreting the p-value
-------------------------

* **p-value ≥ 0.05** ✅: The placebo-effect estimates are not significantly different from zero. This
  is the expected result: when treatment is replaced by noise, the estimator should find no effect.

* **p-value < 0.05** ⚠️: The placebo-effect estimates are significantly different from zero (the
  placebo distribution is not centered at zero). This may indicate that (a) the estimated causal effect
  is spurious, (b) the estimator is not leveraging the true treatment variation, or (c) the sample size
  is too small to reliably detect the effect.

.. note::

   The significance test here is run against a dummy estimate fixed at **0**. So the null hypothesis
   is that the placebo-treatment estimates are consistent with zero. A non-significant p-value means
   the placebo distribution is compatible with zero, while a significant p-value means the placebo
   estimates are systematically away from zero, which is a warning sign for the refutation.

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

