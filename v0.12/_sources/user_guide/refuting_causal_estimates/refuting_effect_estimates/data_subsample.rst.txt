Data Subsample Refuter
======================

This test asks: Does the estimated effect change significantly when we replace the given
dataset with a randomly selected subset? (Hint: It should not)

>>> res_subset=model.refute_estimate(identified_estimand, estimate,
>>>        method_name="data_subset_refuter", show_progress_bar=True, subset_fraction=0.9)
>>> print(res_subset)
