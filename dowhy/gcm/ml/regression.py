from abc import abstractmethod
from typing import Any

import numpy as np
import sklearn
from packaging import version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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

from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.util.general import auto_apply_encoders, auto_fit_encoders, shape_into_2d


class SklearnRegressionModel(PredictionModel):
    """
    General wrapper class for sklearn models.
    """

    def __init__(self, sklearn_mdl: Any) -> None:
        self._sklearn_mdl = sklearn_mdl
        self._encoders = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._encoders = auto_fit_encoders(X, Y)
        X = auto_apply_encoders(X, self._encoders)

        self._sklearn_mdl.fit(X=X, y=Y.squeeze())

    def predict(self, X: np.array) -> np.ndarray:
        return shape_into_2d(self._sklearn_mdl.predict(auto_apply_encoders(X, self._encoders)))

    @property
    def sklearn_model(self) -> Any:
        return self._sklearn_mdl

    def clone(self):
        """
        Clones the prediction model using the same hyper parameters but not fitted.
        :return: An unfitted clone of the prediction model.
        """
        return SklearnRegressionModel(sklearn_mdl=sklearn.clone(self._sklearn_mdl))

    def __str__(self):
        return str(self._sklearn_mdl)


class SklearnRegressionModelWeighted(SklearnRegressionModel):
    def fit(self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray = None) -> None:
        self._encoders = auto_fit_encoders(X, Y)
        X = auto_apply_encoders(X, self._encoders)

        self._sklearn_mdl.fit(X=X, y=Y.squeeze(), sample_weight=sample_weight)


class LinearRegressionWithFixedParameter(PredictionModel):
    def __init__(self, coefficients: np.ndarray, intercept: float):
        self.coefficients = coefficients
        self.intercept = intercept

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.dot(shape_into_2d(X), self.coefficients) + self.intercept).reshape(-1, 1)

    def clone(self):
        return LinearRegressionWithFixedParameter(coefficients=self.coefficients, intercept=self.intercept)


def create_linear_regressor_with_given_parameters(
    coefficients: np.ndarray, intercept: float = 0
) -> LinearRegressionWithFixedParameter:
    return LinearRegressionWithFixedParameter(np.array(coefficients), intercept)


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


def create_polynom_regressor(degree: int = 2, **kwargs_linear_model) -> SklearnRegressionModel:
    return SklearnRegressionModel(
        make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression(**kwargs_linear_model))
    )


class InvertibleFunction:
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Applies the function on the input."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_inverse(self, X: np.ndarray) -> np.ndarray:
        """Returns the outcome of applying the inverse of the function on the inputs."""
        raise NotImplementedError


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
