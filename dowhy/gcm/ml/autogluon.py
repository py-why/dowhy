from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from autogluon import tabular
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from packaging import version

from dowhy.gcm.ml import ClassificationModel, PredictionModel
from dowhy.gcm.util.general import shape_into_2d


class _AutoGluonModel(PredictionModel):
    def __init__(
        self,
        verbosity: int = 0,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        model_persistence_max_memory: Optional[float] = None,
        **fit_parameters,
    ) -> None:
        self._verbosity = verbosity
        self._fit_parameters = fit_parameters
        self._eval_metric = eval_metric
        self._model_persistence_max_memory = model_persistence_max_memory

        self._feature_names = None
        self._target_column = ["Y"]

        self._auto_gluon_model = tabular.TabularPredictor(
            label=self._target_column[0], verbosity=self._verbosity, eval_metric=self._eval_metric, path=path
        )

        if "hyperparameters" not in self._fit_parameters:
            self._fit_parameters["hyperparameters"] = get_hyperparameter_config("light")
            self._fit_parameters["hyperparameters"].update({"LR": {}})

        if "infer_limit" not in self._fit_parameters:
            self._fit_parameters["infer_limit"] = 0.005

        if version.parse(tabular.version.__version__) >= version.parse("0.4"):
            if "presets" not in self._fit_parameters:
                self._fit_parameters["presets"] = ["optimize_for_deployment"]

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = shape_into_2d(X, Y)

        if Y.shape[1] > 1:
            raise RuntimeError("%s supports currently only one dimensional target variables!")

        self._feature_names = ["X" + str(i) for i in range(X.shape[1])]
        self._auto_gluon_model.fit(
            pd.concat(
                [pd.DataFrame(X, columns=self._feature_names), pd.DataFrame(Y, columns=self._target_column)], axis=1
            ),
            **self._fit_parameters,
        )

        if self._model_persistence_max_memory is not None:
            self._auto_gluon_model.persist_models(max_memory=self._model_persistence_max_memory)
        else:
            self._auto_gluon_model.persist_models()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = shape_into_2d(X)
        return shape_into_2d(self._auto_gluon_model.predict(pd.DataFrame(X, columns=self._feature_names)).to_numpy())

    @property
    def auto_gluon_model(self) -> tabular.TabularPredictor:
        return self._auto_gluon_model

    @property
    def fit_parameters(self) -> Dict[str, Any]:
        return self._fit_parameters


class AutoGluonRegressor(_AutoGluonModel):
    def __init__(self, **auto_gluon_parameters) -> None:
        super().__init__(**auto_gluon_parameters)

    def clone(self):
        return AutoGluonRegressor(verbosity=self._verbosity, eval_metric=self._eval_metric, **self.fit_parameters)


class AutoGluonClassifier(_AutoGluonModel, ClassificationModel):
    def __init__(self, **auto_gluon_parameters) -> None:
        super().__init__(**auto_gluon_parameters)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        if self.auto_gluon_model is None:
            raise RuntimeError("AutoGluon model has not been fitted!")

        X = shape_into_2d(X)
        result = self.auto_gluon_model.predict_proba(pd.DataFrame(X, columns=self._feature_names))

        return shape_into_2d(result.to_numpy())

    @property
    def classes(self) -> List[str]:
        return self.auto_gluon_model.class_labels

    def clone(self):
        return AutoGluonClassifier(verbosity=self._verbosity, eval_metric=self._eval_metric, **self.fit_parameters)
