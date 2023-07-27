Estimating conditional average causal effect
============================================

For conditional average causal effect (CACE) estimation, DoWhy relies on the EconML package. All methods from the EconML package can be called from DoWhy's estimation API, thus providing a common interface for estimation methods. In addition, regression methods can be used for CACE estimation too.

For CACE estimation, we need to specify the effect modifiers, either directly while calling `CausalModel` or as a part of the graph. 

Linear regression
-----------------
>>> linear_estimate = model.estimate_effect(identified_estimand, 
>>>                                        method_name="backdoor.linear_regression",
>>>                                       control_value=0,
>>>                                       treatment_value=1)
>>> print(linear_estimate)

DML method from EconML
-----------------------

>>> from sklearn.preprocessing import PolynomialFeatures
>>> from sklearn.linear_model import LassoCV
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> dml_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.dml.DML",
>>>                                     control_value = 0,
>>>                                     treatment_value = 1,
>>>                                 target_units = lambda df: df["X0"]>1,  # condition used for CATE
>>>                                 confidence_intervals=False,
>>>                                method_params={"init_params":{'model_y':GradientBoostingRegressor(),
>>>                                                              'model_t': GradientBoostingRegressor(),
>>>                                                              "model_final":LassoCV(fit_intercept=False), 
>>>                                                              'featurizer':PolynomialFeatures(degree=1, include_bias=False)},
>>>                                               "fit_params":{}})
>> print(dml_estimate)

For a complete analysis using CACE estimators, you can refer to the notebook, :doc:`../../../../example_notebooks/dowhy-conditional-treatment-effects`
