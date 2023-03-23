from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder


class CatBoostEncoder:
    """Implements the proposed method from

    "CatBoost: gradient boosting with categorical features support", Dorogush et al. (2018).

    The Catboost encoder is a target encoder for categorical features. In this implementation we follow Eq. (1)
    in https://arxiv.org/pdf/1810.11363.pdf.
    """

    def __init__(self, p: float = 1, alpha: Optional[float] = None):
        """See Eq. (1) in https://arxiv.org/pdf/1810.11363.pdf

        :param p: The p parameter in the equation. This weights the impact of the given alpha.
        :param alpha: Alpha parameter in the equation. If None is given, the global mean is used as suggested in
                      "A preprocessing scheme for high-cardinality categorical attributes in classification and
                      prediction problems", Micci-Barreca (2001)
        """
        self._p = p
        self._org_alpha = alpha
        self._category_means = None

    def fit(self, X: np.ndarray, Y: np.ndarray, use_alpha_when_unique: bool = True) -> None:
        """Fits the Catboost encoder following https://arxiv.org/pdf/1810.11363.pdf Eq. (1).

        :param X: Input categorical data.
        :param Y: Target data (continuous or categorical)
        :param use_alpha_when_unique: If True, uses the alpha value when a category only appears exactly once.
        """
        self._fit_transform(X, Y, use_alpha_when_unique, train=True)

    def fit_transform(self, X: np.ndarray, Y: np.ndarray, use_alpha_when_unique: bool = True) -> np.ndarray:
        """

        :param X: Input categorical data.
        :param Y: Target data (continuous or categorical).
        :param use_alpha_when_unique: If True, uses the alpha value when a category only appears exactly once.
        :return: Catboost encoded inputs based on the given Y.
        """
        return self._fit_transform(X, Y, use_alpha_when_unique, train=True)

    def transform(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, use_alpha_when_unique: bool = True
    ) -> np.ndarray:
        """Applies the Catboost encoder to the data.

        :param X: Input categorical data.
        :param Y: If target data is given, this data is used instead of the fitted data.
        :param use_alpha_when_unique: If True, uses the alpha value when a category only appears exactly once.
        :return: Catboost encoded inputs. If Y is given, it uses the idea if giving each row a time index and only use
                 the previously observed data to estimate the encoding. If Y is not given, the previously fitted
                 average for each category is used. This can be seen as using the whole training data set as
                 past observations.
        """

        if self._category_means is None:
            raise ValueError("Encoder must be fitted before calling transform")

        if Y is not None:
            return self._fit_transform(X, Y, use_alpha_when_unique, train=False)
        else:
            if X.ndim > 1 and X.shape[1] > 1:
                raise ValueError("CatBoost encoder only supports one dimensional categorical data!")

            X = X.reshape(-1)
            transformed_values = np.zeros(X.shape[0])

            for category in np.unique(X):
                mask = X == category

                if category in self._category_means:
                    transformed_values[mask] = self._category_means[category]
                else:
                    transformed_values[mask] = self._alpha

            return transformed_values

    def _fit_transform(self, X: np.ndarray, Y: np.ndarray, use_alpha_when_unique: bool, train: bool) -> np.ndarray:
        from dowhy.gcm.util.general import is_categorical

        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("CatBoost encoder only supports one dimensional categorical data!")

        if Y.ndim > 1 and Y.shape[1] > 1:
            raise ValueError("CatBoost encoder only supports one dimensional target data!")

        X, Y = X.reshape(-1), Y.reshape(-1)

        if not is_categorical(X):
            raise ValueError("CatBoost encoder only supports categorical input data, i.e., strings!")

        if is_categorical(Y):
            Y = LabelEncoder().fit_transform(Y)

        if train:
            self._alpha = self._org_alpha
            if self._alpha is None:
                self._alpha = np.mean(Y)
            self._category_means = {}

        transformed_values = np.zeros(Y.shape[0])

        for category in np.unique(X):
            mask = X == category
            reduced_Y = Y[mask]

            category_cumulative_sum = np.cumsum(reduced_Y)
            category_cumulative_count = np.cumsum(mask[mask])

            # Eq. (1) in https://arxiv.org/pdf/1810.11363.pdf
            # Subtracting Y here since the cumulative sum includes the current element. The same reason we subtract 1
            # from the count.
            transformed_values[mask] += (category_cumulative_sum - reduced_Y + self._alpha * self._p) / (
                category_cumulative_count + self._p - 1
            )

            if train:
                if use_alpha_when_unique and category_cumulative_count[-1] == 1:
                    self._category_means[category] = self._alpha
                else:
                    self._category_means[category] = (category_cumulative_sum[-1] + self._alpha * self._p) / (
                        category_cumulative_count[-1] + self._p
                    )

        return transformed_values
