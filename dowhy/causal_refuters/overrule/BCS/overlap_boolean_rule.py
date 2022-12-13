#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Dennis Wei, Michael Oberst,         #
#           Fredrik D. Johansson                #
# ----------------------------------------------#

import logging

import cvxpy as cvx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .beam_search import beam_search


class OverlapBooleanRule(object):
    """Overlap Boolean Rule class in the style of scikit-learn"""

    def __init__(
        self,
        alpha=0.9,
        lambda0=1,
        lambda1=1,
        K=10,
        iterMax=100,
        eps=1e-6,
        silent=False,
        verbose=False,
        solver="ECOS",
        D=10,
        logger=None,
        B=5,
        rounding="greedy_sweep",
    ):
        # Fraction of overlap set to cover
        self.alpha = alpha
        # Regularization parameters
        self.lambda0 = lambda0  # clause fixed cost
        self.lambda1 = lambda1  # cost per literal
        # Column generation parameters
        self.K = K  # maximum number of columns generated per iteration
        self.iterMax = iterMax  # maximum number of iterations
        # Numerical tolerance on comparisons
        self.eps = eps
        # Silence output
        self.silent = silent
        if logger is None:
            self.logger = logging.getLogger(__name__)
        self.logger = logger
        # Verbose optimizer
        self.verbose = verbose
        # Solver
        self.solver = solver
        # LP
        self.lp_obj_value = None
        # Maximum Rules considered at each expansion
        self.D = D
        # Rounding
        self.rounding = rounding
        # Beam search width
        self.B = B

        # For get_params / set_params
        self.valid_params = [
            "alpha",
            "lambda0",
            "lambda1",
            "K",
            "iterMax",
            "eps",
            "silent",
            "D",
            "B",
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None

    def fit(self, X, y):
        """Fit model to training data"""
        if not self.silent:
            self.logger.info("Learning Boolean rule set on DNF form with hamming loss")

        # Overlap (y = +1), non-overlap (y = 0), and uniform background (y = -1) samples
        O = np.where(y > 0)[0]
        N = np.where(y == 0)[0]
        U = np.where(y < 0)[0]
        nO = len(O)
        nN = len(N)
        nU = len(U)

        # MKO: We should always have overlap samples, and either background or
        # non-overlap samples
        assert nO > 0 and (nU > 0 or nN > 0)
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
        # print('UB.min():', UB.min())
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

    def greedy_round_(self, X, y, xi=0.5, use_lp=False):
        """
        For DNF, this starts with no conjunctions, and adds them greedily
        based on a cost, which penalizes (any) inclusion of reference samples,
        and rewards (new) inclusion of positive samples, and goes until it
        covers at least alpha fraction of positive samples
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

    def round_(self, X, y, scoring="greedy", xi=None, use_lp=False, tol=0.01):

        """Round based on scoring"""
        if scoring == "greedy":
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
                    if auc > best_auc - tol:
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

    def get_params(self, deep=False):
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
