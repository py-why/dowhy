Estimating Causal Effects
=========================

.. image:: https://raw.githubusercontent.com/microsoft/dowhy/main/docs/images/dowhy-schematic.png
   :alt: Four steps of causal effect estimation in DoWhy
   :width: 100%


The causal effect of a variable $A$ on $Y$ is defined as the expected change in $Y$ due to a change in $A$. Using the do-calculus notation, the average causal effect can be written as, :math:`E[Y|do(A)]`. Sometimes, we are interested in the causal effect only on a subpopulation or want to compare the causal effect across sub-populations. In that case, we can estimate the *conditional* average causal effect (CACE) given a set of covariates $X$, :math:`E[Y|do(A), X]`.

Estimating the causal effect requires four steps: 

1. Model a causal inference problem using assumptions.
2. Identify an expression for the causal effect under these assumptions ("causal estimand").
3. Estimate the expression using statistical methods such as matching or instrumental variables.
4. Finally, verify the validity of the estimate using a variety of robustness checks.

This workflow is captured by four key verbs in DoWhy:

* model (*CausalModel* or *graph*)
* identify (*identify_effect*)
* estimate (*estimate_effect*)
* refute (*refute_estimate*)

Using these verbs, DoWhy implements a causal effect estimation API that supports a variety of methods. *model* encodes prior knowledge as a formal causal graph, *identify* uses graph-based methods to identify the causal effect, *estimate* uses statistical methods for estimating the identified estimand, and finally *refute* tries to refute the obtained estimate by testing robustness to assumptions. Therefore, after building the causal graph, the next step to estimate causal effect is to identify whether the effect can be estimated from available data. In other words,  before considering an estimation algorithm, it is important to determine an identification strategy. DoWhy supports the following identification algorithms:

* Backdoor
* Frontdoor
* Instrumental variable
* ID algorithm

Once a causal effect is identified, we can choose an estimation method compatible with the identification strategy. For estimating the average causal effect, DoWhy supports the following methods.

* Methods based on matching confounders' values:
    * Distance-based matching (:py:class:`DistanceMatchingEstimator <dowhy.causal_estimators.DistanceMatchingEstimator>`)

* Methods based on estimating the treatment assignment
    * Propensity-based Stratification (:py:class:`PropensityScoreStratificationEstimator <dowhy.causal_estimators.PropensityScoreStratificationEstimator>`)
    * Propensity Score Matching (:py:class:`PropensityScoreMatchingEstimator <dowhy.causal_estimators.PropensityScoreMatchingEstimator>`)
    * Inverse Propensity Weighting (:py:class:`PropensityScoreWeightingEstimator <dowhy.causal_estimators.PropensityScoreWeightingEstimator>`)

* Methods based on estimating the outcome model
    * Linear Regression (:py:class:`LinearRegressionEstimator <dowhy.causal_estimators.LinearRegressionEstimator>`)
    * Generalized Linear Models, including logistic regression (:py:class:`GeneralizedLinearModelEstimator <dowhy.causal_estimators.GeneralizedLinearModelEstimator>`)

* Methods based on the instrumental variable equation
    * Binary Instrument/Wald Estimator (:py:class:`InstrumentalVariableEstimator <dowhy.causal_estimators.InstrumentalVariableEstimator>`)
    * Regression discontinuity(:py:class:`RegressionDiscontinuityEstimator <dowhy.causal_estimators.RegressionDiscontinuityEstimator>`)

* Methods for front-door criterion and mediation analysis
    * Two-stage linear regression (:py:class:`TwoStageRegressionEstimator <dowhy.causal_estimators.TwoStageRegressionEstimator>`)


For estimating the conditional average causal effect, DoWhy supports calling EconML methods. For more details on EconML, check out their `documentation <https://econml.azurewebsites.net/>`_.  If the data-generating process for the outcome Y can be approximated as a linear function, you may also use the linear regression method for CACE estimation.

For related notebooks, see :doc:`../../../example_notebooks/nb_index`

.. toctree::
   :maxdepth: 1

   identifying_causal_effect/index
   effect_estimation_with_backdoor/index
   effect_estimation_with_natural_experiments/index
   conditional_effect_estimation/index
   effect_estimation_with_gcm
