ID algorithm for discovering new identification strategies
===========================================================

ID algorithm (Shpitser & Pearl 2006) is an advanced algorithm for identification of causal effect. To use the ID algorithm, you can specify the method name, 


>>> # model is an instance of CausalModel
>>> identified_estimand = model.identify_effect(method_name="id-algorithm")
>>> print(identified_estimand)

Alternatively, you can use the functional API. 

>>> from dowhy.causal_identifier import identify_effect_id
>>> identified_estimand_id = identify_effect_id(
>>>        graph, treatment_name, outcome_name,
>>> )  
>>> # Note that the return type for id_identify_effect is IDExpression and not IdentifiedEstimand  
>>> print(identified_estimand)

To see the ID algorithm in action, check out the example notebook, :doc:`../../../../example_notebooks/identifying_effects_using_id_algorithm`.
