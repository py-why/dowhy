Backdoor criterion
==================

To identify causal effect using the backdoor criterion, we can write, 

>>> # model is an instance of CausalModel
>>> identified_estimand = model.identify_effect()
>>> print(identified_estimand)

The above code uses a default backdoor adjustment. There are four basic kinds of backdoor adjustments available. Each of these are designed to return a valid backdoor set, but they vary in how they select the set of variables to return.

1. *maximal-adjustment*: returns the maximal set that satisfies the backdoor criterion. This is usually the fastest way to find a valid backdoor set, but the set may contain many superfluous variables.
2. *minimal-adjustment*: returns the set with minimal number of variables that satisfies the backdoor criterion. This may take longer to execute, and sometimes may not return any backdoor set within the maximum number of iterations.  
3. *exhaustive-search*: returns all valid backdoor sets. This can take a while to run for large graphs. 
4. *default*: This is a good mix of minimal and maximal adjustment. It starts with maximal adjustment which is usually fast. It then runs minimal adjustment and returns the set having the smallest number of variables.  

To use a specific kind of backdoor adjustment, we can use the `method` argument.

>>> identified_estimand = model.identify_effect(method_name="maximal-adjustment")

Note that the `identify_effect` method is a convenience method that also searches for other kinds of identification. The following is an equivalent call using the functional API. 


>>> from dowhy.causal_identifier import identify_effect_auto, BackdoorAdjustment
>>> identified_estimand_auto = identify_effect_auto(
>>>    graph,
>>>    treatment_name,
>>>    outcome_name,
>>>    backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MAXIMAL
>>> )

