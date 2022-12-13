from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_predict
from xgboost import XGBClassifier

from dowhy.causal_refuters.overrule.ruleset import BCSRulesetEstimator
from dowhy.causal_refuters.overrule.utils import fatom


@dataclass
class SupportConfig:
    """
    Configuration for learning support rules.

    :param n_ref_multiplier: Reference sample count multiplier, defaults to 1.0
    :type n_ref_multiplier: float, optional
    :param alpha: Fraction of the existing examples to ensure are included in the rules, defaults to 0.98
    :type alpha: float, optional
    :param lambda0: Regularization on the # of rules, defaults to 0.0
    :type lambda0: float, optional
    :param lambda1: Regularization on the # of literals, defaults to 0.0
    :type lambda1: float, optional
    :param K: Maximum results returned during beam search, defaults to 20
    :type K: int, optional
    :param D: Maximum extra rules per beam seach iteration, defaults to 20
    :type D: int, optional
    :param B: Width of beam search, defaults to 10
    :type B: int, optional
    :param iterMax: Maximum number of iterations of column generation, defaults to 10
    :type iterMax: int, optional
    :param num_thresh: Number of bins to discretize continuous variables, defaults to 9 (for deciles)
    :type num_thresh: int, optional
    :param solver: Linear programming solver used by CVXPY to solve the LP relaxation, defaults to 'ECOS'
    :type solver: str, optional
    :param rounding: Strategy to perform rounding, either 'greedy' or 'greedy_sweep', defaults to 'greedy_sweep'
    :type rounding: str, optional
    """

    n_ref_multiplier: float = 1
    alpha: float = 0.98
    lambda0: float = 0.0
    lambda1: float = 0.0
    K: int = 20
    D: int = 20
    B: int = 10
    iterMax: int = 10
    num_thresh: int = 9
    solver: str = "ECOS"
    rounding: str = "greedy_sweep"


@dataclass
class OverlapConfig:
    """
    Configuration for learning overlap rules.

    :param alpha: Fraction of the overlap samples to ensure are included in the rules, defaults to 0.95
    :type alpha: float, optional
    :param lambda0: Regularization on the # of rules, defaults to 1e-7
    :type lambda0: float, optional
    :param lambda1: Regularization on the # of literals, defaults to 0.0
    :type lambda1: float, optional
    :param K: Maximum results returned during beam search, defaults to 20
    :type K: int, optional
    :param D: Maximum extra rules per beam seach iteration, defaults to 20
    :type D: int, optional
    :param B: Width of beam search, defaults to 10
    :type B: int, optional
    :param iterMax: Maximum number of iterations of column generation, defaults to 10
    :type iterMax: int, optional
    :param num_thresh: Number of bins to discretize continuous variables, defaults to 9 (for deciles)
    :type num_thresh: int, optional
    :param solver: Linear programming solver used by CVXPY to solve the LP relaxation, defaults to 'ECOS'
    :type solver: str, optional
    :param rounding: Strategy to perform rounding, either 'greedy' or 'greedy_sweep', defaults to 'greedy_sweep'
    :type rounding: str, optional
    """

    alpha: float = 0.95
    lambda0: float = 1e-7
    lambda1: float = 0.0
    K: int = 20
    D: int = 20
    B: int = 10
    iterMax: int = 10
    num_thresh: int = 9
    solver: str = "ECOS"
    rounding: str = "greedy_sweep"

    def __post_init__(self):
        self.n_ref_multiplier = 0.0  # Should not be set to any other value!


class OverruleAnalyzer:
    def __init__(
        self,
        cat_feats: Optional[List[str]] = None,
        support_config: Optional[SupportConfig] = None,
        overlap_config: Optional[OverlapConfig] = None,
        prop_estimator: Optional[Union[BaseEstimator, GridSearchCV]] = None,
        overlap_eps: float = 0.1,
        verbose: bool = False,
    ):
        """
        Learn support and overlap rules.

        :param: cat_feats: List[str]: List of categorical features, all others will be discretized
        :param: support_config: SupportConfig: DataClass with configuration options for learning support rules
        :param: overlap_config: OverlapConfig: DataClass with configuration options for learning overlap rules
        :param: overrule_verbose: bool: Enable verbose logging of optimization output, defaults to False
        :param prop_estimator: Propensity score estimator, defaults to XGBClassifier learned via GridSearchCV
        :type prop_estimator: Optional[Union[BaseEstimator, GridSearchCV]], optional
        :param: overlap_eps: float: Defines the range of propensity scores for a point to be considered in the overlap
            region, with the range defined as `(overlap_eps, 1 - overlap_eps)`, defaults to 0.1
        :param verbose: Verbose optimization output, defaults to False
        :type verbose: bool, optional
        """
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
        self.is_fitted = False

    def fit(self, X, t):
        # Do the support characterization
        self.RS_support_estimator.fit(X)
        # Recover the samples that are in the support
        supp = self.RS_support_estimator.predict(X).astype(bool)
        X_supp, t_supp = X[supp], t[supp]

        # Get the propensity scores out. Note that we perform cross-fitting here
        self.raw_overlap_set = self._predict_overlap(X_supp, t_supp)
        self.RS_overlap_estimator.fit(X_supp, self.raw_overlap_set)

        self.support_indicator = supp
        self.overlap_indicator = self.RS_overlap_estimator.predict(X_supp)
        self.is_fitted = True
        self.X = X
        self.t = t
        self.X_supp = X_supp
        self.t_supp = t_supp

    def _predict_overlap(self, X, t):
        prop_scores = cross_val_predict(self.prop_estimator, X, t.values.ravel(), method="predict_proba")
        prop_scores = prop_scores[:, 1]  # Probability of treatment
        overlap_set = np.logical_and(prop_scores < 1 - self.overlap_eps, prop_scores > self.overlap_eps)
        return overlap_set

    def predict(self, X):
        self._check_is_fitted()
        supp_ind = self.RS_support_estimator.predict(X)
        overlap_ind = self.RS_overlap_estimator.predict(X)
        return supp_ind * overlap_ind

    def describe_all_rules(self):
        self._check_is_fitted()
        coverage = self.predict(self.X).mean()
        return_str = "SUMMARY:\n"
        return_str = f"Rules cover {coverage:.1%} of all samples\n"
        return_str += (
            f"Overall, {self.raw_overlap_set.mean():.1%} of samples meet the criteria for inclusion in the overlap set, "
            "defined as: Covered by support rules and propensity score in "
            f"({self.overlap_eps:.2f}, {1 - self.overlap_eps:.2f})\n"
        )
        true_positive = self.overlap_indicator * self.raw_overlap_set
        overlap_coverage = true_positive.sum() / self.raw_overlap_set.sum()
        return_str += f"Rules capture {overlap_coverage:.1%} of samples which meet these criteria\n"
        return_str += "\nDETAILED RULES:\n"
        return_str += self.describe_support_rules()
        return_str += self.describe_overlap_rules()
        return return_str

    def describe_support_rules(self):
        self._check_is_fitted()
        X = self.X
        s_est = self.RS_support_estimator
        return self._describe_rules(s_est, X, estimator_name="SUPPORT")

    def describe_overlap_rules(self):
        self._check_is_fitted()
        X = self.X_supp
        o_est = self.RS_overlap_estimator
        return self._describe_rules(o_est, X, estimator_name="OVERLAP")

    def _describe_rules(self, estimator, X, estimator_name=""):
        rules_by_sample = estimator.predict_rules(X)
        rules_active = estimator.M.w
        active_rules_by_sample = rules_by_sample[:, rules_active.astype(bool)]
        sample_coverage = active_rules_by_sample.mean(axis=0).tolist()
        rule_list = []
        for r in zip(estimator.rules(as_str=False), sample_coverage):
            rule_list.append({"rule": r[0], "coverage": r[1]})

        # For DNF rules, a sample is covered if *any* rule applies
        total_coverage = active_rules_by_sample.max(axis=1).mean()

        if estimator_name == "Overlap":
            return_str = (
                f"{estimator_name} Rules: Found {len(rule_list)} rule(s), "
                f"covering {total_coverage:.1%} of samples in the Support set\n"
            )
        else:
            return_str = (
                f"{estimator_name} Rules: Found {len(rule_list)} rule(s), covering {total_coverage:.1%} of samples\n"
            )
        for idx, r in enumerate(rule_list):
            if idx == 0:
                prefix = "   "
            else:
                prefix = "OR "

            return_str += f"\t {prefix}Rule #{idx}: "
            return_str += self._print_rule(r["rule"])
            return_str += f"\t\t [Covers {r['coverage']:.1%} of samples]\n"

        return return_str

    def _print_rule(self, rule):
        return_str = ""
        for idx, a in enumerate(rule):
            if idx == 0:
                prefix = ""
            else:
                prefix = "\t\t AND "
            rule_str = fatom(a[0], a[1], a[2])
            return_str += f"{prefix}({rule_str})\n"

        return return_str

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise ValueError("Call .fit() before describing rules")

    def __str__(self):
        return self.describe_all_rules()