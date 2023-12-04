Regression-based methods
========================

Linear regression is one of the most common methods to estimate causal effect. It is useful when the data-generating process for the outcome Y can be approximated as a linear function. 

Given a backdoor identified estimand, to estimate causal effect using linear regression, we can write, 

>>> estimate = model.estimate_effect(identified_estimand,
>>>        method_name="backdoor.linear_regression",
>>>        test_significance=True
>>> )
>>> print(estimate)

The above method combines fitting the model and estimating the causal effect. To obtain more control, we can use the functional API.

>>> # Fit the regression estimator
>>> estimator = LinearRegressionEstimator(
>>>    identified_estimand=identified_estimand,
>>>    test_significance=True,
>>> ).fit(
>>>    data=data["df"],
>>>    effect_modifier_names=graph.get_effect_modifiers(treatment_name, outcome_name),
>>> )
>>> # Estimate the effect given treatment and control value
>>> estimate = estimator.estimate_effect(
>>>    data=data["df"],
>>>    control_value=0,
>>>    treatment_value=1,
>>>    target_units="ate",
>>> )

In addition to linear regression, DoWhy supports generalized linear models. This can be used to fit a logistic regression model.


