from dataclasses import asdict, dataclass
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_predict
from xgboost import XGBClassifier

from dowhy.causal_refuters.overrule.ruleset import BCSRulesetEstimator


@dataclass
class SupportConfig:
    n_ref_multiplier: float = 1
    alpha: float = 0.98
    lambda0: float = 0.0
    lambda1: float = 0.0
    K: int = 20  # Maximum results returned during beam search
    D: int = 20  # Maximum extra rules per beam seach iteration
    B: int = 10  # Width of Beam Search
    iterMax: int = 10
    num_thresh: int = 5
    solver: str = "ECOS"
    rounding: str = "greedy_sweep"


@dataclass
class OverlapConfig:
    n_ref_multiplier: float = 0.0
    alpha: float = 0.95
    lambda0: float = 1e-7
    lambda1: float = 0.0
    K: int = 20  # Maximum results returned during beam search
    D: int = 20  # Maximum extra rules per beam seach iteration
    B: int = 10  # Width of Beam Search
    iterMax: int = 10
    num_thresh: int = 5
    solver: str = "ECOS"
    rounding: str = "greedy_sweep"


class OverruleAnalyzer:
    def __init__(
        self,
        cat_feats: List[str],
        prop_estimator=None,
        overlap_eps=0.1,
        support_config=None,
        overlap_config=None,
        verbose=False,
    ):
        if support_config is None:
            support_config = SupportConfig()
        if overlap_config is None:
            overlap_config = OverlapConfig()

        if not isinstance(support_config, SupportConfig):
            raise ValueError("support_config not a SupportConfig class")

        if not isinstance(overlap_config, OverlapConfig):
            raise ValueError("overlap_config not a OverlapConfig class")

        self.RS_support_estimator = BCSRulesetEstimator(
            cat_cols=cat_feats, silent=verbose, verbose=verbose, **asdict(support_config)
        )
        self.RS_overlap_estimator = BCSRulesetEstimator(
            cat_cols=cat_feats, silent=verbose, verbose=verbose, **asdict(overlap_config)
        )
        self.overlap_eps = overlap_eps

        if prop_estimator is None:
            param_grid = {"max_depth": [2, 4, 6], "n_estimators": [200]}
            estimator = XGBClassifier(objective="binary:logistic", random_state=0)
            prop_estimator = GridSearchCV(estimator=estimator, param_grid=param_grid)

        if not isinstance(prop_estimator, BaseEstimator) and not isinstance(prop_estimator, GridSearchCV):
            raise ValueError("Propensity estimator is not an sklearn estimator")

        self.prop_estimator = prop_estimator

    def fit(self, X, t):
        # Do the support characterization
        self.RS_support_estimator.fit(X)
        # Recover the samples that are in the support
        supp = self.RS_support_estimator.predict(X).astype(bool)
        X_supp, t_supp = X[supp], t[supp]

        # Get the propensity scores out. Note that we perform cross-fitting here
        prop_scores = cross_val_predict(self.prop_estimator, X_supp, t_supp.values.ravel(), method="predict_proba")
        prop_scores = prop_scores[:, 1]  # Probability of treatment
        overlap_indicator = np.logical_and(prop_scores < 1 - self.overlap_eps, prop_scores > self.overlap_eps)

        self.RS_overlap_estimator.fit(X_supp, overlap_indicator)
        self.s_rules = self.RS_support_estimator.rules(as_str=False)
        self.o_rules = self.RS_overlap_estimator.rules(as_str=False)
