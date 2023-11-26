Mediation Analysis: Estimating natural direct and indirect effects
==================================================================

Mediation analysis can be used to quantify the extent to which a causal influence is exerted through a specific pathway. DoWhy supports the estimation of the *natural direct effect* and the *natural indirect effect*:

**Natural direct effect**: Effect due to the path v0->y
**Natural indirect effect**: Effect due to the path v0->FD0->y (mediated by FD0).

For more details, see `Interpretation and Identification of Causal Mediation <https://ftp.cs.ucla.edu/pub/stat_ser/r389-imai-etal-commentary-r421-reprint.pdf>`_ by Judea Pearl.

Using DoWhy's effect estimation framework, we can perform a mediation analysis by adjusting the estimand_type argument accordingly:

Identification 
--------------

>>> # Natural direct effect (nde)
>>> identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde", 
>>>                                                 proceed_when_unidentifiable=True)
>>> print(identified_estimand_nde)

>>> # Natural indirect effect (nie)
>>> identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie",
>>>                                                 proceed_when_unidentifiable=True)
>>> print(identified_estimand_nie)

Estimation
----------

>>> import dowhy.causal_estimators.linear_regression_estimator
>>> causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
>>>                                             method_name="mediation.two_stage_regression",
>>>                                             confidence_intervals=False,
>>>                                             test_significance=False,
>>>                                             method_params = {
>>>                                                 'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
>>>                                                 'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
>>>                                             })
>>> print(causal_estimate_nie)

>>> causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
>>>                                             method_name="mediation.two_stage_regression",
>>>                                             confidence_intervals=False,
>>>                                             test_significance=False,
>>>                                             method_params = {
>>>                                               'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
>>>                                               'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
>>>                                             })
>>> print(causal_estimate_nde)

Related example notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`../../../example_notebooks/dowhy_mediation_analysis`