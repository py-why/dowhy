"""This module defines implementations of :class:`~dowhy.gcm.ml.PredictionModel` used by the different
:class:`~dowhy.gcm.graph.FunctionalCausalModel` implementations, such as :class:`~dowhy.gcm.causal_mechanisms.PostNonlinearModel` or
:class:`~dowhy.gcm.causal_mechanisms.AdditiveNoiseModel`.
"""

from .classification import (
    ClassificationModel,
    SklearnClassificationModel,
    create_gaussian_process_classifier,
    create_hist_gradient_boost_classifier,
    create_logistic_regression_classifier,
    create_polynom_logistic_regression_classifier,
    create_random_forest_classifier,
)
from .prediction_model import PredictionModel
from .regression import (
    InvertibleFunction,
    SklearnRegressionModel,
    create_elastic_net_regressor,
    create_gaussian_process_regressor,
    create_hist_gradient_boost_regressor,
    create_lasso_lars_ic_regressor,
    create_lasso_regressor,
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_polynom_regressor,
    create_random_forest_regressor,
    create_ridge_regressor,
    create_support_vector_regressor,
)

try:
    from dowhy.gcm.ml.autogluon import AutoGluonClassifier, AutoGluonRegressor
except ImportError:
    pass
