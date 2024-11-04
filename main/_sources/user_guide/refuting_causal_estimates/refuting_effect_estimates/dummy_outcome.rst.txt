Dummy Outcome Refuter
=====================

The dummy outcome refuter tests: What happens to the estimated causal effect when we replace the true outcome
variable with an independent random variable? (Hint: The effect should go to zero)
In addition, an extension of the test can also check for any simulated outcome where the causal effect need not be zero: What happens to the estimated causal effect when we replace the outcome with
a simulated outcome based on a known data-generating process closest to the given dataset? (Hint: It
should match the effect parameter from the data-generating process)


Testing for zero causal effect
-------------------------------
>>> ref = model.refute_estimate(identified_estimand,
>>>                           causal_estimate,
>>>                           method_name="dummy_outcome_refuter"
>>>                           )
>>> print(ref[0])

Testing for non-zero causal effect
----------------------------------

>>> coefficients = np.array([1,2])
>>> bias = 3
>>> def linear_gen(df):
>>>     y_new = np.dot(df[['W0','W1']].values,coefficients) + 3
>>>     return y_new

>>> ref = model.refute_estimate(identified_estimand,
>>>                           causal_estimate,
>>>                           method_name="dummy_outcome_refuter",
>>>                           outcome_function=linear_gen
>>>                           )
>>> print(ref[0])


For a complete example on using the dummy outcome refuter, you can check out the notebook, :doc:`../../../../example_notebooks/dowhy_demo_dummy_outcome_refuter`.
