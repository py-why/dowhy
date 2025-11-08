Natural experiments and instrumental variables
==============================================


To identify effect using the instrumental variable criterion, we use the same `identify_effect` method. It checks for all possible identification strategies among backdoor, frontdoor, and instrumental variables. 

>>> # model is an instance of CausalModel
>>> identified_estimand = model.identify_effect()
>>> print(identified_estimand)

For an example of using instrumental variable strategy to estimate causal effect, see :doc:`../../../../example_notebooks/dowhy-simple-iv-example`.
