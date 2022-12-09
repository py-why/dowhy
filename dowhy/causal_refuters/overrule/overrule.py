# -----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets   #
# @Authors: Fredrik D. Johansson, Michael Oberst #
# -----------------------------------------------#

import inspect

import numpy as np
import pandas as pd
from sklearn import base, model_selection
from sklearn.metrics import roc_auc_score

from . import overlap, ruleset, support
from .overlap import OverlapEstimator, SupportOverlapEstimator
from .ruleset import BCSRulesetEstimator, RulesetEstimator
from .support import SupportEstimator, SVMSupportEstimator


class OverRule(OverlapEstimator):
    """Overlap Estimation using Rule Sets"""

    def __init__(
        self,
        overlap_estimator="support",
        support_estimator=SVMSupportEstimator,
        support_estimator_1=None,
        alpha_s=0.1,
        alpha_r=0.05,
        n_ref_multiplier=1.0,
        support_kwargs={},
        ruleset_kwargs={},
        ruleset_estimator="bcs",
    ):
        """Initializes the overlap characterization model.

        @args
            overlap_estimator: The class of estimator used to estimate
                overlap. Should be either an instance of a subclass to
                OverlapEstimator or a string 'support'
            support_estimator: The class of estimator used for estimation of
                group-wise support. Either a string in ['svm',...] or an
                implementation of support.SupportEstimator.
            support_estimator_1: Specific support estimator for class 1.
                Defaults to support_estimator if None is supplied.
            alpha_s: The maximum amount of support to leave out
            alpha_r: The maximum amount of the overlap set leave out
            n_ref_multiplier: A multiplier for the number of reference samples.
                The total number is n*dim*multiplier.
            support_kwargs: Keyword arguments for support estimators
            ruleset_kwargs: Keyword arguments for ruleset estimators
            ruleset_estimator: Type of ruleset estimator. Should be an instance
                of RulesetEstimator or a string in ['bcs', 'dt']
        """

        # Initialize overlap estimator
        if overlap_estimator == "support":
            # Estimator based on separate support estimation
            self.O = SupportOverlapEstimator(
                support_estimator,
                alpha_s,
                support_estimator_1=support_estimator_1,
                **support_kwargs
            )
        elif isinstance(overlap_estimator, overlap.OverlapEstimator):
            # Estimator based on instance of OverlapEstimator
            self.O = overlap_estimator
        else:
            # Unknown argument
            raise Exception("Unknown overlap estimator type")

        # Initialize ruleset Estimator (see RulesetEstimator class)
        self.alpha_r = alpha_r
        self._init_ruleset_estimator(
            ruleset_estimator,
            alpha_r,
            n_ref_multiplier=n_ref_multiplier,
            **ruleset_kwargs
        )

    def fit(self, x, g):
        """Given samples (x_i, g_i) of features x_i and (binary) group
            membership g_i, this method fits an Overlap Ruleset by first
            estimating the supports s_0 and s_1 of the conditionals p(x|g_i=0)
            and p(x|g_i=1) and later fitting an interpretable model of the
            support indicator s_0*s_1.
        @args
            x: features
            g: group indicators, assumed to be binary
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        g = (
            g.values.ravel()
            if isinstance(g, pd.DataFrame) or isinstance(g, pd.Series)
            else g.ravel()
        )

        self.O.fit(X, g)
        o = self.O.predict(X)

        self.RS.fit(X, o)

    def predict(self, x, use_density=False):
        """Predicts whether a given point is a member of the overlap set,
        optionally using the underlying density estimator
        @args
            x: Test point
            use_density: Use the density estimators instead of the simple rule
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        if use_density:
            return self.O.predict(X)
        else:
            return self.RS.predict(X)

    def predict_rules(self, x):
        """Predicts whether a given point satisfies the rules,
        optionally using the underlying density estimator
        @args
            x: Test point
            use_density: Use the density estimators instead of the simple rule
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        return self.RS.predict_rules(X)

    def rules(self, as_str=False, transform=None, fmt="%.3f", labels={}):
        """Returns rules learned by the estimator

        @args
            as_str: Returns a string if True, otherwise a dictionary
            transform: A function that takes key-value pairs for rules and
                thresholds and transforms the value. Used to re-scale
                standardized data for example
            fmt: Formatting string for float values
        """

        if isinstance(self.RS, model_selection.GridSearchCV):
            try:
                M = self.RS.best_estimator_
            except:
                raise Exception("Grid search not performed yet")
        else:
            M = self.RS

        return M.rules(as_str=as_str, transform=transform, fmt=fmt, labels=labels)

    def complexity(self):
        """Returns number of rules and number of atoms"""
        rules = self.rules()

        n_rules = len(rules)
        n_atoms = np.sum([len(r) for r in rules])

        return n_rules, n_atoms

    def round(self, x, scoring="roc_auc"):
        """Round rule set"""
        o = self.predict(x, use_density=True)
        self.predict(x)

        self.RS.round_(x, o, scoring=scoring)

    def score(self, x, o):
        """Evaluates the fitted models. If a label o is supplied, the score
            measures the accuracy of the overlap estimation.
        @args
            x: Test point
            o: Label indicating overlap status at x
        """
        return roc_auc_score(o, self.predict(x))

    def score_vs_base(self, x):
        """Evaluates the rule set approximation of the base estimator
        @args
            x: Test point
            o: Label indicating overlap status at x (optional)
        """
        return roc_auc_score(self.O.predict(x), self.predict(x))

    def _init_ruleset_estimator(self, ruleset_estimator, alpha, **kwargs):
        """Initializes ruleset estimator, denoted RS

        @args
            ruleset_estimator: The class of estimator used for estimation of
                binary rules to characterize the overlap region
            alpha: The maximum fraction of the overlap set to leave out.
                Does not override parameters if instance is given for
                ruleset_estimator (latter takes precedence).
            kwargs: Additional keyword arguments
        """

        if inspect.isclass(ruleset_estimator) and issubclass(
            ruleset_estimator, RulesetEstimator
        ):
            self.RS = ruleset_estimator(**kwargs)
        elif isinstance(ruleset_estimator, RulesetEstimator) or isinstance(
            ruleset_estimator, base.BaseEstimator
        ):
            self.RS = ruleset_estimator
        elif ruleset_estimator == "bcs":
            self.RS = BCSRulesetEstimator(alpha=1 - alpha, **kwargs)
        else:
            raise Exception("Unknown rule set estimator type")

    def get_params(self, deep=False):
        raise Exception("Not implemented.")

    def set_params(self, **params):
        raise Exception("Not implemented.")


class OverRule2Stage(OverlapEstimator):
    """Overlap Estimation using Rule Sets"""

    def __init__(
        self,
        overlap_estimator,
        overlap_rs_estimator,
        support_rs_estimator,
        bb_on_support=False,
        refit_s=True,
        refit_o=True,
    ):
        """Initializes the overlap characterization model.

        @args
            overlap_estimator: Should be an instance of a subclass to OverlapEstimator
            overlap_rs_estimator: An implementation of RulesetEstimator
            support_rs_estimator: An implementation of RulesetEstimator
            bb_on_support: Use only predicted support samples for learning black-box
        """

        self.bb_on_support = bb_on_support
        self.refit_s = refit_s
        self.refit_o = refit_o

        # Initialize overlap estimator
        if isinstance(overlap_estimator, overlap.OverlapEstimator):
            self.O = overlap_estimator
        else:
            raise Exception("Only overlap estimators of type OverlapEstimator")

        # Initialize overlap ruleset estimator
        if isinstance(overlap_rs_estimator, RulesetEstimator):
            self.RS_o = overlap_rs_estimator
            self.RS_o.set_params(
                gamma=0, n_ref_multiplier=0
            )  # Make sure negative samples not used
        else:
            raise Exception("Only overlap ruleset estimators of type RulesetEstimator")

        # Initialize support ruleset estimator
        if isinstance(support_rs_estimator, RulesetEstimator):
            self.RS_s = support_rs_estimator
        else:
            raise Exception("Only support ruleset estimators of type RulesetEstimator")

    def fit(self, x, g):
        """Given samples (x_i, g_i) of features x_i and (binary) group
            membership g_i, this method fits an Overlap Ruleset by first
            estimating the supports s_0 and s_1 of the conditionals p(x|g_i=0)
            and p(x|g_i=1) and later fitting an interpretable model of the
            support indicator s_0*s_1.
        @args
            x: features
            g: group indicators, assumed to be binary
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        g = (
            g.values.ravel()
            if isinstance(g, pd.DataFrame) or isinstance(g, pd.Series)
            else g.ravel()
        )

        if self.refit_s:
            self.RS_s.fit(X, np.ones(X.shape[0]))

        I_s = self.RS_s.predict(X) > 0

        if self.refit_o:
            if self.bb_on_support:
                self.O.fit(X.iloc[I_s], g[I_s])
            else:
                self.O.fit(X, g)

        o = self.O.predict(X)

        self.RS_o.fit(X.iloc[I_s], o[I_s])

        self.RS_o.round_(X.iloc[I_s], o[I_s], "greedy_sweep")

    def predict(self, x, use_density=False, support_only=True):
        """Predicts whether a given point is a member of the overlap set,
        optionally using the underlying density estimator
        @args
            x: Test point
            use_density: Use the density estimators instead of the simple rule
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        supp_mul = 1.0
        if support_only:
            supp_mul = self.RS_s.predict(X)

        if use_density:
            return self.O.predict(X) * supp_mul
        else:
            return self.RS_o.predict(X) * supp_mul

    def predict_support(self, x):
        """Predicts whether a given point is in the support
        @args
            x: Test point
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        return self.RS_s.predict(X)

    def predict_rules(self, x):
        """Predicts whether a given point satisfies the rules,
        optionally using the underlying density estimator
        @args
            x: Test point
            use_density: Use the density estimators instead of the simple rule
        """

        X = (
            x
            if isinstance(x, pd.DataFrame)
            else pd.DataFrame(dict([("x%d" % i, x[:, i]) for i in range(x.shape[1])]))
        )

        return self.RS_o.predict_rules(X) * self.RS_s.predict_rules(X)

    def rules(self, as_str=False, transform=None, fmt="%.3f", labels={}):
        """Returns rules learned by the estimator

        @args
            as_str: Returns a string if True, otherwise a dictionary
            transform: A function that takes key-value pairs for rules and
                thresholds and transforms the value. Used to re-scale
                standardized data for example
            fmt: Formatting string for float values
        """

        if isinstance(self.RS_s, model_selection.GridSearchCV):
            try:
                M_s = self.RS_s.best_estimator_
            except:
                raise Exception("Grid search not performed yet")
        else:
            M_s = self.RS_s

        if isinstance(self.RS_o, model_selection.GridSearchCV):
            try:
                M_o = self.RS_o.best_estimator_
            except:
                raise Exception("Grid search not performed yet")
        else:
            M_o = self.RS_o

        return (
            M_s.rules(as_str=as_str, transform=transform, fmt=fmt, labels=labels),
            M_o.rules(as_str=as_str, transform=transform, fmt=fmt, labels=labels),
        )

    def complexity(self):
        """Returns number of rules and number of atoms"""
        rules_s, rules_o = self.rules()

        n_rules_s = len(rules_s)
        n_atoms_s = np.sum([len(r) for r in rules_s])
        n_rules_o = len(rules_o)
        n_atoms_o = np.sum([len(r) for r in rules_o])

        return n_rules_s, n_atoms_s, n_rules_o, n_atoms_o

    def round(self, x, scoring="roc_auc"):
        """Round rule set"""
        o = self.predict(x, use_density=True)

        self.RS_o.round_(x, o, scoring=scoring)

    def score(self, x, o):
        """Evaluates the fitted models. If a label o is supplied, the score
            measures the accuracy of the overlap estimation.
        @args
            x: Test point
            o: Label indicating overlap status at x
        """
        return roc_auc_score(o, self.predict(x))

    def score_vs_base(self, x):
        """Evaluates the rule set approximation of the base estimator
        @args
            x: Test point
            o: Label indicating overlap status at x (optional)
        """
        return roc_auc_score(self.O.predict(x), self.predict(x))

    def get_params(self, deep=False):
        raise Exception("Not implemented.")

    def set_params(self, **params):
        raise Exception("Not implemented.")
