"""Code for Binarizing Features.

This module implements the boolean ruleset estimator from OverRule [1]. Code is adapted (with some simplifications)
from https://github.com/clinicalml/overlap-code, under the MIT License.

[1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class FeatureBinarizer(TransformerMixin):
    """
    Transformer for binarizing categorical and ordinal (including continuous) features.

    Note that all features are converted into binary variables before learning Boolean rules.
    """

    def __init__(
        self,
        colCateg: List[str] = [],
        numThresh: int = 9,
        negations: bool = False,
        threshStr: bool = False,
        threshOverride: Dict = {},
        **kwargs,
    ):
        """
        Initialize transformer for binarizing categorical and ordinal (including continuous) features

        :param colCateg: List of categorical columns, defaults to [], 'object' dtype automatically treated as categorical
        :type colCateg: List[str], optional
        :param numThresh: Number of quantile thresholds to binarize ordinal features, defaults to 9
        :type numThresh: int, optional
        :param negations: Include negations, defaults to False
        :type negations: bool, optional
        :param threshStr: Convert thresholds to strings, defaults to False
        :type threshStr: bool, optional
        :param threshOverride: Dictionary to override quantile thresholds, defaults to {},
            formatted as `{colname : np.linspace object}` to define cuts
        :type threshOverride: Dict, optional
        """
        # List of categorical columns
        if type(colCateg) is pd.Series:
            self.colCateg = colCateg.tolist()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg

        self.threshOverride = {} if threshOverride is None else threshOverride
        # Number of quantile thresholds used to binarize ordinal features
        self.numThresh = numThresh
        self.thresh: Dict[str, np.ndarray] = {}
        # whether to append negations
        self.negations = negations
        # whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr

    def fit(self, X):
        """
        Fit to data, including the learning of thresholds where appropriate.

        Sets the following internal variables:
        * `maps` = dictionary of mappings for unary/binary columns
        * `enc` = dictionary of OneHotEncoders for categorical columns
        * `thresh` = dictionary of lists of thresholds for ordinal columns
        * `NaN` = list of ordinal columns containing NaN values

        :param X: Original features as a Pandas Dataframe
        :type X: pd.DataFrame
        """
        data = X
        # Quantile probabilities
        quantProb = np.linspace(1.0 / (self.numThresh + 1.0), self.numThresh / (self.numThresh + 1.0), self.numThresh)
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        NaN = []

        # Iterate over columns
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(data[c].unique()))

            # Categorical column
            elif (c in self.colCateg) or (data[c].dtype == "object"):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore")
                # Fit to observed categories
                enc[c].fit(data[[c]])

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) | np.issubdtype(data[c].dtype, np.dtype(float).type):
                # Few unique values
                if valUniq <= self.numThresh + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                elif c in self.threshOverride.keys():
                    thresh[c] = self.threshOverride[c]
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
                if data[c].isnull().any():
                    # Contains NaN values
                    NaN.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.NaN = NaN
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data into binary features.

        :param X: Original features as a Pandas Dataframe
        :type X: pd.DataFrame
        :return A: Binary feature dataframe
        :type A: pd.DataFrame
        """
        data = X
        maps = self.maps
        enc = self.enc
        thresh = self.thresh
        NaN = self.NaN

        # Initialize dataframe
        A = pd.DataFrame(
            index=data.index, columns=pd.MultiIndex.from_arrays([[], [], []], names=["feature", "operation", "value"])
        )

        # Iterate over columns
        for c in data:
            # Constant or binary column
            if c in maps:
                # Rename values to 0, 1
                A[(str(c), "", "")] = data[c].map(maps[c])
                if self.negations:
                    A[(str(c), "not", "")] = 1 - A[(str(c), "", "")]

            # Categorical column
            elif c in enc:
                # Apply OneHotEncoder
                Anew = enc[c].transform(data[[c]])
                Anew = pd.DataFrame(Anew, index=data.index, columns=enc[c].categories_[0].astype(str))
                if self.negations:
                    # Append negations
                    Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(str(c), "=="), (str(c), "!=")])
                else:
                    Anew.columns = pd.MultiIndex.from_product([[str(c)], ["=="], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif c in thresh:
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                if self.negations:
                    # Append negations
                    Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                    ops = ["<=", ">"]
                else:
                    ops = ["<="]
                # Convert to dataframe with column labels
                if self.threshStr:
                    Anew = pd.DataFrame(
                        Anew,
                        index=data.index,
                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]),
                    )
                else:
                    Anew = pd.DataFrame(
                        Anew, index=data.index, columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]])
                    )
                if c in NaN:
                    # Ensure that rows corresponding to NaN values are zeroed out
                    indNull = data[c].isnull()
                    Anew.loc[indNull] = 0
                    # Add NaN indicator column
                    Anew[(str(c), "==", "NaN")] = indNull.astype(int)
                    if self.negations:
                        Anew[(str(c), "!=", "NaN")] = (~indNull).astype(int)
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        return A
