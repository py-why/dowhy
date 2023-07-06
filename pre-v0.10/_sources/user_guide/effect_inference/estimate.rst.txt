Estimate causal effect based on the identified estimand
------------------------------------------------------------

DoWhy supports methods based on both back-door criterion and instrumental
variables. It also provides a non-parametric confidence intervals and a permutation test for testing
the statistical significance of obtained estimate.

Supported estimation methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Methods based on estimating the treatment assignment
    * Propensity-based Stratification
    * Propensity Score Matching
    * Inverse Propensity Weighting

* Methods based on estimating the outcome model
    * Linear Regression
    * Generalized Linear Models

* Methods based on the instrumental variable equation
    * Binary Instrument/Wald Estimator
    * Two-stage least squares
    * Regression discontinuity

* Methods for front-door criterion and general mediation
    * Two-stage linear regression

Examples of using these methods are in the `Estimation methods
<https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy_estimation_methods.ipynb>`_
notebook.

Using EconML and CausalML estimation methods in DoWhy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is easy to call external estimation methods using DoWhy. Currently we
support integrations with the `EconML <https://github.com/microsoft/econml>`_ and `CausalML <https://github.com/uber/causalml>`_ packages. Here's an example
of estimating conditional treatment effects using EconML's double machine
learning estimator.

.. code:: python

	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.linear_model import LassoCV
	from sklearn.ensemble import GradientBoostingRegressor
	dml_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.dml.DML",
                        control_value = 0,
                        treatment_value = 1,
                        target_units = lambda df: df["X0"]>1,
                        confidence_intervals=False,
                        method_params={
                            "init_params":{'model_y':GradientBoostingRegressor(),
                                           'model_t': GradientBoostingRegressor(),
                                           'model_final':LassoCV(),
                                           'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            "fit_params":{}}
						)


More examples are in the `Conditional Treatment Effects with DoWhy
<https://github.com/microsoft/dowhy/blob/main/docs/source/example_notebooks/dowhy-conditional-treatment-effects.ipynb>`_ notebook.


