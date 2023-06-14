"""OverlapBooleanRule.

This module implements the boolean ruleset estimator from OverRule [1]. Code is adapted (with some simplifications)
from https://github.com/clinicalml/overlap-code, under the MIT License.

[1] Oberst, M., Johansson, F., Wei, D., Gao, T., Brat, G., Sontag, D., & Varshney, K. (2020). Characterization of
Overlap in Observational Studies. In S. Chiappa & R. Calandra (Eds.), Proceedings of the Twenty Third International
Conference on Artificial Intelligence and Statistics (Vol. 108, pp. 788–798). PMLR. https://arxiv.org/abs/1907.04138
"""

import logging
from typing import Union

import cvxpy as cvx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .beam_search import beam_search


class OverlapBooleanRule:
    """Overlap Boolean Rule class in the style of scikit-learn"""

    def __init__(
        self,
        alpha=0.95,
        lambda0=1e-2,
        lambda1=1e-2,
        K=20,
        D=20,
        B=10,
        iterMax=10,
        eps=1e-6,
        silent=False,
        verbose=False,
        solver="ECOS",
        rounding="greedy_sweep",
    ):
        """
        Learn Boolean Rules in Disjuntive Normal Form to describe the positive class.

        :param alpha: Fraction of the positive samples to ensure are included in the rules, defaults to 0.95
        :type alpha: float, optional
        :param lambda0: Regularization on the # of rules, defaults to 1e-2
        :type lambda0: float, optional
        :param lambda1: Regularization on the # of literals, defaults to 1e-2
        :type lambda1: float, optional
        :param K: Maximum results returned during beam search, defaults to 20
        :type K: int, optional
        :param D: Maximum extra rules per beam seach iteration, defaults to 20
        :type D: int, optional
        :param B: Width of beam search, defaults to 10
        :type B: int, optional
        :param iterMax: Maximum number of iterations of column generation, defaults to 10
        :type iterMax: int, optional
        :param eps: Numerical tolerance on comparisons, defaults to 1e-6
        :type eps: float, optional
        :param silent: Silence non-optimizer output, defaults to False
        :type silent: bool
        :param verbose: Verbose optimizer output, defaults to False
        :type verbose: bool, optional
        :param solver: Linear programming solver used by CVXPY to solve the LP relaxation, defaults to 'ECOS'
        :type solver: str, optional
        :param rounding: Strategy to perform rounding, either 'greedy' or 'greedy_sweep', defaults to 'greedy_sweep'
        :type rounding: str, optional
        """
        self.alpha = alpha
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.K = K
        self.D = D
        self.B = B
        self.iterMax = iterMax
        self.eps = eps
        self.silent = silent
        self.verbose = verbose
        self.solver = solver
        self.rounding = rounding

        self.logger = logging.getLogger(__name__)
        self.lp_obj_value = None

        # For get_params / set_params
        self.valid_params = [
            "alpha",
            "lambda0",
            "lambda1",
            "K",
            "D",
            "B",
            "iterMax",
            "silent",
            "eps",
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.DataFrame]):
        """
        Fit model to training data.

        :param X: Pandas DataFrame containing covariates
        :param y: +1 for Overlap/Support (depending on rules being learned), 0 for
            non-overlap, and -1 for background samples.  Should only contain (+1/0) for
            overlap rules, or (+1/-1) for learning support rules.
        """
        if not self.silent:
            self.logger.info("Learning Boolean rule set on DNF form with hamming loss")

        # Overlap / Support (y = +1), non-overlap (y = 0), and uniform background (y = -1) samples
        O = np.where(y > 0)[0]
        N = np.where(y == 0)[0]
        U = np.where(y < 0)[0]
        nO = len(O)
        nN = len(N)
        nU = len(U)

        # We should always have overlap samples, and either background or non-overlap samples
        # This will throw an error if, for example, all samples are considered to
        # be in the overlap region
        assert nO > 0 and (
            nU > 0 or nN > 0
        ), "Recieved positive samples, but no negative samples for learning Boolean Rules"
        assert nU == 0 or nN == 0

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1] + 1, 1, dtype=int), index=X.columns)
        A = np.hstack((np.ones((X.shape[0], 1), dtype=int), X))
        # Iteration counter
        self.it = 0
        # Formulate master LP
        # Variables
        w = cvx.Variable(A.shape[1], nonneg=True)
        xiO = cvx.Variable(nO, nonneg=True)
        # Objective function (no penalty on empty conjunction)
        lambdas = self.lambda0 + self.lambda1 * z.sum().values
        lambdas[0] = 0
        if nU:
            obj = cvx.Minimize(cvx.sum(A[U, :] @ w) / nU + lambdas @ w)
        elif nN:
            obj = cvx.Minimize(cvx.sum(A[N, :] @ w) / nN + lambdas @ w)
        # Constraints
        # This gets activated for DNF
        constraints = [cvx.sum(xiO) <= (1 - self.alpha) * nO, xiO + A[O, :] @ w >= 1]

        # Solve problem
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=self.verbose, solver=self.solver)

        # Extract dual variables
        r = np.zeros_like(y, dtype=float)
        r[O] = -constraints[1].dual_value
        if nU:
            r[U] = 1.0 / nU
        elif nN:
            r[N] = 1.0 / nN

        if not self.silent:
            self.logger.info("Initial solve completed")

        # Beam search for conjunctions with negative reduced cost
        # Most negative reduced cost among current variables
        UB = np.dot(r, A) + lambdas
        UB = min(UB.min(), 0)
        v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1, K=self.K, UB=UB, eps=self.eps, B=self.B, D=self.D)

        while (v < -self.eps).any() and (self.it < self.iterMax):
            # Negative reduced costs found
            self.it += 1

            if not self.silent:
                self.logger.info("Iteration: %d, Objective: %.4f" % (self.it, prob.value))

            # Add to existing conjunctions
            z = pd.concat([z, zNew], axis=1, ignore_index=True)
            A = np.concatenate((A, Anew), axis=1)

            # Reformulate master LP
            # Variables
            w = cvx.Variable(A.shape[1], nonneg=True)
            # Objective function
            lambdas = np.concatenate((lambdas, self.lambda0 + self.lambda1 * zNew.sum().values))
            if nU:
                obj = cvx.Minimize(cvx.sum(A[U, :] @ w) / nU + lambdas @ w)
            elif nN:
                obj = cvx.Minimize(cvx.sum(A[N, :] @ w) / nN + lambdas @ w)
            # Constraints
            constraints = [cvx.sum(xiO) <= (1 - self.alpha) * nO, xiO + A[O, :] @ w >= 1]
            # Solve problem
            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose=self.verbose, solver=self.solver)

            # Extract dual variables
            r = np.zeros_like(y, dtype=float)
            r[O] = -constraints[1].dual_value
            if nU:
                r[U] = 1.0 / nU
            elif nN:
                r[N] = 1.0 / nN

            # Beam search for conjunctions with negative reduced cost
            # Most negative reduced cost among current variables
            UB = np.dot(r, A) + lambdas
            # print('UB.min():', UB.min())
            UB = min(UB.min(), 0)
            v, zNew, Anew = beam_search(
                r, X, self.lambda0, self.lambda1, K=self.K, UB=UB, eps=self.eps, B=self.B, D=self.D
            )

        # Save generated conjunctions and coefficients
        self.z = z
        w = w.value

        self.w_raw = w
        self.lp_obj_value = prob.value

        self.round_(X, y, scoring=self.rounding)

    def greedy_round_(self, X: pd.DataFrame, y: Union[np.ndarray, pd.DataFrame], xi: float = 0.5, use_lp: bool = False):
        """
        Round the rule coefficients to integer values.

        For DNF, this starts with no conjunctions, and adds them greedily based on a cost, which penalizes (any)
        inclusion of negative samples, and rewards (new) inclusion of positive samples, and goes until it covers at
        least alpha fraction of positive samples.

        :param X: Pandas DataFrame containing covariates
        :param y: +1 for Overlap/Support (depending on rules being learned), 0 for
            non-overlap, and -1 for background samples.  Should only contain (+1/0) for
            overlap rules, or (+1/-1) for learning support rules.
        :param xi: Reward for including positive samples, relative to cost (1) for including negative samples
        :param use_lp: Restrict to those conjuctions where the LP coefficients are positive.  Note that the LP makes a
            difference regardless, as we only consider the rules generated by column generation here.
        """

        A = self.compute_conjunctions(X)
        R = np.arange(0, A.shape[1])  # Remaining conjunctions
        U = []  # Used conjunctions
        C = np.zeros(X.shape[0])  # Coverage indicator
        MAX_ITER = 1000
        i = 0

        # Restrict conjunctions to those used by LP
        if use_lp:
            R = [R[i] for i in range(len(R)) if self.w_raw[i] > 0]

        # NOTE: This is a greedy approach, so it does not incorporate lambda0 explicitly
        # Similarly, it will prefer a larger number of smaller rules if lambda1 is set
        # to a larger value, because the incremental cost will be lower.
        while (i < MAX_ITER) and (C[y == 1].mean() < self.alpha):
            if (y == 0).sum() > 0:
                neg_cover = (A[(y == 0), :][:, R]).mean(0)
            else:
                neg_cover = 0

            if (y == -1).sum() > 0:
                # Fraction of reference samples that each conjunction covers
                ref_cover = (A[(y == -1), :][:, R]).mean(0)
            else:
                ref_cover = 0

            # Regularization (for each conjunction)
            reg = self.lambda1 * self.z.values[:, R].sum(0)

            # Positive samples newly covered (for each conjunction)
            pos_new_cover = (A[(y == 1) & (C < 1), :][:, R] + 1e-8).mean(0)

            # Costs (for each conjunction)
            costs = neg_cover + ref_cover + reg - xi * pos_new_cover

            r = np.argmin(costs)  # Find min-cost conjunction
            C = (C + A[:, R[r]]) > 0.0  # Update coverage
            U.append(R[r])
            R = np.array([R[i] for i in range(len(R)) if not i == r])

            i += 1

        # Zero out the rules and only take those which are used
        self.w = np.zeros(A.shape[1])
        self.w[U] = 1

    def round_(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.DataFrame],
        scoring: str = "greedy",
        xi=None,
        use_lp: bool = True,
    ):
        """
        Round the rule coefficients to integer values via a greedy approach, either using a fixed reward
        (`scoring="greedy"`) or optimizing the reward for including positive examples according
        to balanced accuracy on classifying positive vs negative samples (`scoring="greedy_sweep`).

        :param X: Pandas DataFrame containing covariates
        :param y: +1 for Overlap/Support (depending on rules being learned), 0 for
            non-overlap, and -1 for background samples.  Should only contain (+1/0) for
            overlap rules, or (+1/-1) for learning support rules.
        :param xi: Reward for including positive samples, relative to cost (1) for including negative samples. For
            `scoring="greedy"`, should be a single value, or an array of values for `scoring="greedy_sweep"`.  For the
            latter, will default to `np.logspace(np.log10(0.01), 0.5, 20)`.
        :param use_lp: Restrict to those conjuctions where the LP coefficients are positive.  Note that the LP makes a
            difference regardless, as we only consider the rules generated by column generation here.
        """
        if scoring == "greedy":
            if xi is None:
                xi = 0.5
            self.greedy_round_(X, y, xi=xi, use_lp=use_lp)

        elif scoring == "greedy_sweep":
            if xi is None:
                xi = np.logspace(np.log10(0.01), 0.5, 20)
            xis = np.array([xi]).ravel()
            best_xi = xis[0]
            if len(xis) > 1:
                best_xi = None
                best_auc = 0
                for xii in xis:
                    self.greedy_round_(X, y, xi=xii, use_lp=use_lp)
                    auc = roc_auc_score(y, self.predict(X))
                    # Small tolerance on comparisons
                    # This can be useful to break ties and favor larger values of xi
                    if auc > best_auc - 0.001:
                        best_xi = xii
                    if auc > best_auc:
                        best_auc = auc
            self.greedy_round_(X, y, xi=best_xi, use_lp=use_lp)

    def get_objective_value(self, X, o, rounded=True):
        if rounded:
            w = self.w
        else:
            w = self.w_raw

        U = np.where(o < 0)[0]
        nU = len(U)
        assert nU > 0

        A = self.compute_conjunctions(X)
        lambdas = self.lambda0 + self.lambda1 * self.z.sum().values
        lambdas[0] = 0

        obj = np.sum(A[U, :].dot(w)) / nU + lambdas.dot(w)

        return obj

    def compute_conjunctions(self, X):
        """Compute conjunctions of features specified in self.z"""
        try:
            A = 1 - (np.dot(1 - X, self.z) > 0)
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        return A

    def predict_(self, X, w):
        """Predict whether points belong to overlap region"""
        # Compute conjunctions of features
        A = self.compute_conjunctions(X)
        # Predict labels
        return (np.dot(A, w) > 0).astype(int)

    def predict(self, X):
        """Predict whether points belong to overlap region"""
        # Use helper function
        return self.predict_(X, self.w)

    def predict_rules(self, X):
        """Predict whether points belong to overlap region"""
        # Use helper function
        A = self.compute_conjunctions(X)

        return ((A * self.w) > 0).astype(int)

    def get_params(self):
        """Returns estimator parameters"""
        return dict([(k, getattr(self, k)) for k in self.valid_params])

    def set_params(self, **params):
        """Sets estimator parameters"""
        if not params:
            return self

        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)

        return self
