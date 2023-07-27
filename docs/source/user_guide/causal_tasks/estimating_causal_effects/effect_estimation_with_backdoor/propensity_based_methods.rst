Propensity-based methods
========================

Propensity-based methods are backdoor estimation methods that involve estimating the action as a function of the backdoor variables, :math:`P(A|W)`. This fitted function is then used to derive matching, stratification or weighting methods. 

Propensity-Based Matching
-------------------------

>>> causal_estimate_match = model.estimate_effect(identified_estimand,
>>>                                              method_name="backdoor.propensity_score_matching",
>>>                                              target_units="atc")
>>> print(causal_estimate_match)
>>> print("Causal Estimate is " + str(causal_estimate_match.value))

Propensity-based Stratification
--------------------------------

>>> causal_estimate_strat = model.estimate_effect(identified_estimand,
>>>                                              method_name="backdoor.propensity_score_stratification",
>>>                                              target_units="att")
>>> print(causal_estimate_strat)
>>> print("Causal Estimate is " + str(causal_estimate_strat.value))

Inverse Propensity Weighting
----------------------------

>>> causal_estimate_ipw = model.estimate_effect(identified_estimand,
>>>                                            method_name="backdoor.propensity_score_weighting",
>>>                                            target_units = "ate",
>>>                                            method_params={"weighting_scheme":"ips_weight"})
>>> print(causal_estimate_ipw)
>>> print("Causal Estimate is " + str(causal_estimate_ipw.value))


