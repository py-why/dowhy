from abc import abstractmethod
from typing import List

import numpy as np
import sklearn
from packaging import version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from dowhy.gcm.ml.prediction_model import PredictionModel

if version.parse(sklearn.__version__) < version.parse("1.0"):
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dowhy.gcm.ml.regression import SklearnRegressionModel, SklearnRegressionModelWeighted
from dowhy.gcm.util.general import auto_apply_encoders, shape_into_2d


class ClassificationModel(PredictionModel):
    @abstractmethod
    def predict_probabilities(self, X: np.array) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        raise NotImplementedError


class SklearnClassificationModel(SklearnRegressionModel, ClassificationModel):
    def predict_probabilities(self, X: np.array) -> np.ndarray:
        return shape_into_2d(self._sklearn_mdl.predict_proba(auto_apply_encoders(X, self._encoders)))

    @property
    def classes(self) -> List[str]:
        return self._sklearn_mdl.classes_

    def clone(self):
        return SklearnClassificationModel(sklearn_mdl=sklearn.clone(self._sklearn_mdl))


class SklearnClassificationModelWeighted(SklearnRegressionModelWeighted, ClassificationModel):
    def predict_probabilities(self, X: np.array) -> np.ndarray:
        return shape_into_2d(self._sklearn_mdl.predict_proba(auto_apply_encoders(X, self._encoders)))

    @property
    def classes(self) -> List[str]:
        return self._sklearn_mdl.classes_

    def clone(self):
        return SklearnClassificationModel(sklearn_mdl=sklearn.clone(self._sklearn_mdl))


def create_random_forest_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(RandomForestClassifier(**kwargs))


def create_gaussian_process_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(GaussianProcessClassifier(**kwargs))


def create_hist_gradient_boost_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(HistGradientBoostingClassifier(**kwargs))


def create_logistic_regression_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(LogisticRegression(**kwargs))


def create_extra_trees_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(ExtraTreesClassifier(**kwargs))


def create_ada_boost_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(AdaBoostClassifier(**kwargs))


def create_support_vector_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(SVC(**kwargs, probability=True))


def create_knn_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(KNeighborsClassifier(**kwargs))


def create_gaussian_nb_classifier(**kwargs) -> SklearnClassificationModel:
    return SklearnClassificationModel(GaussianNB(**kwargs))


def create_polynom_logistic_regression_classifier(
    degree: int = 3, **kwargs_logistic_regression
) -> SklearnClassificationModel:
    return SklearnClassificationModel(
        make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False), LogisticRegression(**kwargs_logistic_regression)
        )
    )
