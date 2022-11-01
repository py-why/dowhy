"""Functions and classes in this module should be considered experimental, meaning there might be breaking API changes
in the future.
"""

from typing import Any

import numpy as np
import sklearn
from packaging import version

if version.parse(sklearn.__version__) < version.parse("1.0"):
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LassoLarsIC, LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from dowhy.gcm.fcms import InvertibleFunction, PredictionModel
from dowhy.gcm.util.general import apply_one_hot_encoding, fit_one_hot_encoders, shape_into_2d


class SklearnRegressionModel(PredictionModel):
    """
    General wrapper class for sklearn models.
    """

    def __init__(self, sklearn_mdl: Any) -> None:
        self._sklearn_mdl = sklearn_mdl
        self._one_hot_encoders = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._one_hot_encoders = fit_one_hot_encoders(X)
        X = apply_one_hot_encoding(X, self._one_hot_encoders)

        self._sklearn_mdl.fit(X=X, y=Y.squeeze())

    def predict(self, X: np.array) -> np.ndarray:
        return shape_into_2d(self._sklearn_mdl.predict(apply_one_hot_encoding(X, self._one_hot_encoders)))

    @property
    def sklearn_model(self) -> Any:
        return self._sklearn_mdl

    def clone(self):
        """
        Clones the prediction model using the same hyper parameters but not fitted.

        :return: An unfitted clone of the prediction model.
        """
        return SklearnRegressionModel(sklearn_mdl=sklearn.clone(self._sklearn_mdl))


def create_linear_regressor_with_given_parameters(
    coefficients: np.ndarray, intercept: float = 0, **kwargs
) -> SklearnRegressionModel:
    linear_model = LinearRegression(**kwargs)
    linear_model.coef_ = coefficients
    linear_model.intercept_ = intercept

    return SklearnRegressionModel(linear_model)


def create_linear_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(LinearRegression(**kwargs))


def create_ridge_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(RidgeCV(**kwargs))


def create_lasso_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(LassoCV(**kwargs))


def create_lasso_lars_ic_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(LassoLarsIC(**kwargs))


def create_elastic_net_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(ElasticNetCV(**kwargs))


def create_gaussian_process_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GaussianProcessRegressor(**kwargs))


def create_support_vector_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(SVR(**kwargs))


def create_random_forest_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(RandomForestRegressor(**kwargs))


def create_hist_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(HistGradientBoostingRegressor(**kwargs))


def create_extra_trees_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(ExtraTreesRegressor(**kwargs))


def create_knn_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(KNeighborsRegressor(**kwargs))


def create_ada_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(AdaBoostRegressor(**kwargs))


def create_product_regressor() -> PredictionModel:
    return ProductRegressor()


class InvertibleIdentityFunction(InvertibleFunction):
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return X

    def evaluate_inverse(self, X: np.ndarray) -> np.ndarray:
        return X


class InvertibleExponentialFunction(InvertibleFunction):
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X)

    def evaluate_inverse(self, X: np.ndarray) -> np.ndarray:
        return np.log(X)


class InvertibleLogarithmicFunction(InvertibleFunction):
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.log(X)

    def evaluate_inverse(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X)


class ProductRegressor(PredictionModel):
    def __init__(self):
        self._one_hot_encoders = {}

    def fit(self, X, Y):
        self._one_hot_encoders = fit_one_hot_encoders(X)

    def predict(self, X):
        X = apply_one_hot_encoding(X, self._one_hot_encoders)
        return np.prod(X, axis=1).reshape(-1, 1)

    def clone(self):
        return ProductRegressor()
