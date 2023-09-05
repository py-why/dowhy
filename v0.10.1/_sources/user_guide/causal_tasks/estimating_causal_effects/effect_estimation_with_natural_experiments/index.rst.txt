Estimating average causal effect with natural experiments
=========================================================


Natural experiments include situations where an instrumental variable affects the action or there is a regression discontinuity. The method of instrumental variables is a common identifying strategy for natural experiments. Below we show two estimators derived from this identification strategy. 


Instrumental variable estimator for binary action
-------------------------------------------------

>>> causal_estimate_iv = model.estimate_effect(identified_estimand,
>>>        method_name="iv.instrumental_variable", method_params = {'iv_instrument_name': 'Z0'})
>>> print(causal_estimate_iv)
>>> print("Causal Estimate is " + str(causal_estimate_iv.value))

Regression discontinuity estimator
-----------------------------------

>>> causal_estimate_regdist = model.estimate_effect(identified_estimand,
>>>        method_name="iv.regression_discontinuity", 
>>>        method_params={'rd_variable_name':'Z1',
>>>                       'rd_threshold_value':0.5,
>>>                       'rd_bandwidth': 0.15})
>>> print(causal_estimate_regdist)
>>> print("Causal Estimate is " + str(causal_estimate_regdist.value))
