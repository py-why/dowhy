from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
import sklearn
from packaging import version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

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

    def fit_dataframe(self, X: pd.DataFrame, Y: np.ndarray) -> None:
        """Fits the model using a pandas DataFrame, storing column names for column-order-safe inference.

        By default converts X to a NumPy array and delegates to :meth:`fit`. Override in subclasses that
        need to retain the original pandas schema (e.g. wrappers for AutoML frameworks).

        :param X: Feature DataFrame whose columns are the parent node names in fitting order.
        :param Y: Target labels.
        """
        self._feature_names = list(X.columns)
        self.fit(X.to_numpy(), Y)

    def predict_probabilities_dataframe(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities, automatically reordering DataFrame columns to match the fitting order.

        This is a safer alternative to :meth:`predict_probabilities` when the caller cannot guarantee
        that columns are provided in the same order as during fitting.  Requires that the model was
        previously fitted via :meth:`fit_dataframe`.

        :param X: Feature DataFrame containing at least the columns seen during fitting.
        :return: A nxd numpy matrix of class probabilities.
        :raises ValueError: If the model was not fitted with ``fit_dataframe`` or required columns are missing.
        """
        if not hasattr(self, "_feature_names"):
            raise ValueError(
                "This model was not fitted with fit_dataframe(). "
                "Call fit_dataframe() first to enable column-name-aware inference."
            )
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing columns that were present during fitting: {missing}. "
                f"Expected columns (in order): {self._feature_names}."
            )
        return self.predict_probabilities(X[self._feature_names].to_numpy())


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
        return SklearnClassificationModelWeighted(sklearn_mdl=sklearn.clone(self._sklearn_mdl))


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


def create_decision_tree_classifier() -> SklearnClassificationModel:
    return SklearnClassificationModel(DecisionTreeClassifier())
