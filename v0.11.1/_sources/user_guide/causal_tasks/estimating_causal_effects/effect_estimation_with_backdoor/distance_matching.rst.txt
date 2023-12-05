Distance-based matching
========================

>>> causal_estimate_dmatch = model.estimate_effect(identified_estimand,
>>>                                              method_name="backdoor.distance_matching",
>>>                                              target_units="att",
>>>                                              method_params={'distance_metric':"minkowski", 'p':2})
>>> print(causal_estimate_dmatch)
