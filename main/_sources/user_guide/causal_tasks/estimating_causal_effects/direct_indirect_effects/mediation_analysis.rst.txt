Estimating natural direct and indirect effects
==============================================

We use the estimand_type argument to specify that the target estimand should be for a natural direct effect or the natural indirect effect. For definitions, see `Interpretation and Identification of Causal Mediation <https://ftp.cs.ucla.edu/pub/stat_ser/r389-imai-etal-commentary-r421-reprint.pdf>`_ by Judea Pearl.

**Natural direct effect**: Effect due to the path v0->y

**Natural indirect effect**: Effect due to the path v0->FD0->y (mediated by FD0).

Identification 
--------------

>>> # Natural direct effect (nde)
>>> identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde", 
>>>                                            proceed_when_unidentifiable=True)
>>> print(identified_estimand_nde)

>>> # Natural indirect effect (nie)
>>> identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie",
>>>                                            proceed_when_unidentifiable=True)
>>> print(identified_estimand_nie)

Estimation
----------

>>> import dowhy.causal_estimators.linear_regression_estimator
>>> causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
>>>                                        method_name="mediation.two_stage_regression",
>>>                                       confidence_intervals=False,
>>>                                       test_significance=False,
>>>                                        method_params = {
>>>                                            'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
>>>                                            'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
>>>                                        }
>>>                                       )
>>> print(causal_estimate_nie)

>>> causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
>>>                                        method_name="mediation.two_stage_regression",
>>>                                       confidence_intervals=False,
>>>                                       test_significance=False,
>>>                                        method_params = {
>>>                                            'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
>>>                                            'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
>>>                                        }
>>>                                       )
>>> print(causal_estimate_nde)

