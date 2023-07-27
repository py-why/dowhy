Frontdoor criterion
===================

To identify effect using the frontdoor criterion, we use the same `identify_effect` method. It checks for all possible identification strategies among backdoor, frontdoor, and instrumental variables. 

>>> # model is an instance of CausalModel
>>> identified_estimand = model.identify_effect()
>>> print(identified_estimand)


