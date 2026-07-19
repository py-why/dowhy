from abc import abstractmethod

import numpy as np
import pandas as pd


class PredictionModel:
    """Represents general prediction model implementations. Each prediction model should provide a fit and a predict
    method."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    def fit_dataframe(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """Fits the model using a pandas DataFrame and Series, preserving column names and dtypes.

        By default this converts X and Y to NumPy arrays and delegates to :meth:`fit`. Override in
        subclasses (e.g. AutoGluon wrappers) that need to retain the original pandas schema—column
        names, categorical dtypes, etc.—for correct downstream behaviour.

        :param X: Feature DataFrame whose columns are the ordered predecessor nodes.
        :param Y: Target Series.
        """
        self.fit(X.to_numpy(), Y.to_numpy())

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        """
        Clones the prediction model using the same hyper parameters but not fitted.

        :return: An unfitted clone of the prediction model.
        """
        raise NotImplementedError
