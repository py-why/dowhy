Random Common Cause Refuter
===========================

The Add Random Common Cause refuter checks the following: Does the estimation method change its estimate after we add an independent random variable as a common cause to the dataset? (Hint: It should not)


>>> res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", show_progress_bar=True)
>>> print(res_random)
