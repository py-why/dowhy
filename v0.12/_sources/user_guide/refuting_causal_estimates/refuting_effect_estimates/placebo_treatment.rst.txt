Placebo Treatment Refuter
=========================

Placebo Treatment refutation asks: What happens to the estimated causal effect when we replace the true treatment
variable with an independent random variable? (Hint: the effect should go to zero)

>>> res_placebo=model.refute_estimate(identified_estimand, estimate,
>>>        method_name="placebo_treatment_refuter", show_progress_bar=True, placebo_type="permute")
>>> print(res_placebo)

