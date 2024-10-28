"""Ruleset estimator class for OverRule.

This module implements the boolean ruleset estimator from OverRule [1]. Code is adapted (with some simplifications)
from https://github.com/clinicalml/overlap-code, under the MIT License.

[1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .BCS.load_process_data_BCS import FeatureBinarizer
from .BCS.overlap_boolean_rule import OverlapBooleanRule
from .utils import rule_str, sample_reference


class BCSRulesetEstimator:
    """Ruleset estimator based on Boolean Rules with Column Generation.

    Operates according to an scikit-learn interface with a few additional methods.
    """

    def __init__(
        self,
        n_ref_multiplier: float = 1.0,
        lambda0: float = 0.0,
        lambda1: float = 0.0,
        cat_cols: Optional[List] = None,
        negations: bool = True,
        num_thresh: int = 9,
        seed: int = None,
        ref_range: Optional[Dict[str, Dict]] = None,
        thresh_override: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initializes the estimator.

        `**kwargs` are passed to OverlapBooleanRule (see ./BCS/overlap_boolean_rule.py for description of arguments)

        :param n_ref_multiplier: Reference sample count multiplier, only used for estimating support, defaults to 1.0,
            but should be set to zero for Overlap rules
        :type n_ref_multiplier: float, optional
        :param lambda0: Regularization on the # of rules, defaults to 0.0
        :type lambda0: float, optional
        :param lambda1: Regularization on the # of literals, defaults to 0.0
        :type lambda1: float, optional
        :param cat_cols: Set of categorical columns, defaults to None
        :type cat_cols: Optional[List], optional
        :param negations: Include negation of literals, defaults to True
        :type negations: bool, optional
        :param num_thresh: Number of bins to discretize continuous variables, defaults to 9 (for deciles)
        :type num_thresh: int, optional
        :param seed: Random seed for reference samples, only used for estimating support, defaults to None
        :type seed: int, optional
        :param ref_range: Manual override of the range for reference samples, given as a dictionary of the form
            `ref_range = {c: {"is_binary": True/False, "min": min_value, "max": max_value}}`
        :type ref_range: Optional[Dict], optional
        :param thresh_override: Manual override of the thresholds for continuous features, given as a dictionary like
            the following, will only be applied to continuous features with more than num_thresh unique values
            `thresh_override = {column_name: np.linspace(0, 100, 10)}`
        :type thresh_override: Optional[Dict], optional
        """
        # Parameters
        self.n_ref_multiplier = n_ref_multiplier
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.cat_cols = cat_cols if cat_cols else []
        self.negations = negations
        self.num_thresh = num_thresh
        self.seed = seed
        self.ref_range = ref_range
        self.thresh_override = thresh_override
        self.kwargs = kwargs

        # Bookkeeping
        self.refSamples = None
        self.overlapSamples = None

        # Initialize estimators
        self.init_estimator_()

        self.valid_params = ["lambda0", "lambda1", "cat_cols", "n_ref_multiplier", "negations", "num_thresh", "seed"]

    def __getstate__(self):
        state = self.__dict__.copy()
        if "logger" in self.kwargs.keys():
            state["kwargs"]["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def init_estimator_(self):
        """Initialize rule set estimator and feature binarizer."""
        self.M = OverlapBooleanRule(lambda0=self.lambda0, lambda1=self.lambda1, **self.kwargs)

        self.FeatureBinarizer = FeatureBinarizer(
            negations=self.negations,
            colCateg=self.cat_cols,
            numThresh=self.num_thresh,
            threshOverride=self.thresh_override,
        )

    def fit(self, x, o=None):
        """
        Fit rules for either characterizing support (if O is not provided) or for characterizing overlap, in which case
        O should be a vector indicating overlap by 1 and non-overlap by 0.

        This function is primarily a wrapper around the OverlapBooleanRule estimator, making sure that features are
        binarized before being fed into the ruleset estimator, constructing reference samples for the support
        characterization, and so on.

        :param x: Samples of covariates
        :type x: Pandas DataFrame or Numpy Array, shape (n, d)
        :param o: Binary indicator for whether or not a sample belongs in the overlap region, defaults to None.  If
            provided, should have the same length as `x`
        :type o: Pandas DataFrame or Numpy Array, shape (n, )
        """

        n = x.shape[0]
        dim = x.shape[1]
        nRef = int(n * dim * self.n_ref_multiplier)
        if o is None:
            o = np.ones((n,))

        # Convert to dataframe if not
        X = x if isinstance(x, pd.DataFrame) else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))

        # Format labels
        o = o.values.ravel() if (isinstance(o, pd.DataFrame) or isinstance(o, pd.Series)) else o.ravel()

        # Sample from reference measure and construct features
        self.refSamples = sample_reference(X, n=nRef, seed=self.seed, ref_range=self.ref_range)

        # Add reference samples
        data = pd.concat([X, self.refSamples], axis=0, sort=False)
        o = np.hstack([o, -np.ones(nRef)])

        # Binarize features (fit to data only)
        self.FeatureBinarizer.fit(data.iloc[:n])
        X = self.FeatureBinarizer.transform(data)

        # Fit estimator
        self.M.fit(X, o)

        # Store reference volume
        if nRef > 0:
            self.relative_volume = self.predict(self.refSamples).mean()

        return self

    def predict(self, x):
        """
        Predict whether or not X lies in the overlap region (1 = True).

        :param x: Samples of covariates
        :type x: Pandas DataFrame or Numpy Array, shape (n, d)
        """
        # Construct features dataframe
        data = (
            x if isinstance(x, pd.DataFrame) else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        X = self.FeatureBinarizer.transform(data).fillna(0)

        preds = self.M.predict(X)

        return preds

    def predict_rules(self, x):
        """
        Predict rules activated by x

        :param x: Samples of covariates
        :type x: Pandas DataFrame or Numpy Array, shape (n, d)
        :return: Matrix with binary values, of shape (n, r), where r is the total number of rules considered by the
            estimator, and where 1 indicates that the sample matches the rule, and 0 indicates otherwise.
        :rtype: Numpy Array, shape (n, r)
        """
        # Construct features dataframe
        data = (
            x if isinstance(x, pd.DataFrame) else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        X = self.FeatureBinarizer.transform(data).fillna(0)

        return self.M.predict_rules(X)

    def rules(
        self,
        as_str: bool = False,
        transform: Optional[Callable[[str, float], float]] = None,
        fmt: str = "%.3f",
        labels: Dict[str, str] = {},
    ):
        """
        Return rules learned by the estimator.

        :param as_str: Return a string if True, otherwise a dictionary, defaults to False
        :type as_str: bool, optional
        :param transform: A function that takes key-value pairs for rules and thresholds and transforms the value.
            This function is used to re-scale standardized data, defaults to None
        :type transform: Optional[Callable[[str, float], float]], optional
        :param fmt: Formatting string for float values, for printing rules with thresholds, defaults to "%.3f"
        :type fmt: str, optional
        :param labels: Dictionary mapping from original feature names to display names when printing rules, any
            feature not specified here will default to the original name, defaults to {}
        :type labels: Dict[str, str], optional
        """
        w, z = (self.M.w, self.M.z)
        w_sel = np.where(w)[0]

        def t_(k, v):
            return v if transform is None else transform(k, v)

        C = []
        for j in w_sel:
            index_j = z[z[j] == 1][j].index
            f = index_j.get_level_values(0).values
            o = index_j.get_level_values(1).values
            v = index_j.get_level_values(2).values

            l = [labels.get(a, a) for a in f]

            dis_j = [(l[i], o[i], t_(f[i], v[i])) for i in range(len(f))]
            C.append(dis_j)

        if as_str:
            return rule_str(C, fmt=fmt)
        else:
            return C

    def get_params(self, deep=False):
        """Return estimator parameters"""
        params = dict([(k, getattr(self, k)) for k in self.valid_params])

        if deep:
            return {**params, **self.M.get_params(deep=True)}
        else:
            return params

    def set_params(self, **params):
        """Set estimator parameters"""
        if not params:
            return self

        reinit = False
        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)
            elif k in self.M.valid_params:
                reinit = True
                self.kwargs[k] = v
        if reinit:
            self.init_estimator_()

        return self
