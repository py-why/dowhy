#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Dennis Wei, Michael Oberst,         #
#           Fredrik D. Johansson                #
# ----------------------------------------------#

import logging
import os
import numpy as np
import pandas as pd
from .load_process_data_BCS import extract_target, binarize_features
import cvxpy as cvx
from .beam_search import beam_search, beam_search_no_dup

from sklearn.metrics import roc_auc_score

class OverlapBooleanRule(object):
    """Overlap Boolean Rule class in the style of scikit-learn"""
    def __init__(self, alpha=0.9, hamming=True, gamma=1, lambda0=1, lambda1=1, K=10, \
                 iterMax=100, eps=1e-6, silent=False, CNF=False, verbose=False, solver='ECOS', D=10,
                 logger=None, B=5,
                 rounding='coverage'):
        # Fraction of overlap set to cover
        self.alpha = alpha
        # Use Hamming loss instead of 0-1 loss
        self.hamming = hamming
        # Relative weight on uniform background samples
        self.gamma = gamma
        # Regularization parameters
        self.lambda0 = lambda0      # clause fixed cost
        self.lambda1 = lambda1      # cost per literal
        # Column generation parameters
        self.K = K                  # maximum number of columns generated per iteration
        self.iterMax = iterMax      # maximum number of iterations
        # Numerical tolerance on comparisons
        self.eps = eps
        # Silence output
        self.silent = silent
        if logger is None:
            logger = logging.getLogger('OverlapBooleanRule')
        self.logger = logger
        # CNF instead of DNF (NOTE: CNF=True and hamming=False not supported)
        self.CNF = CNF
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
        # @TODO: Maybe make this class variable?
        self.valid_params = ['alpha', 'hamming', 'gamma', 'lambda0',
                             'lambda1', 'K', 'iterMax', 'eps',
                             'silent', 'CNF', 'D', 'B']

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None

    def fit(self, X, y):
        """Fit model to training data"""
        if not self.silent:
            self.logger.info('Learning Boolean rule set on %s form %s hamming loss' % ('CNF' if self.CNF else 'DNF', 'with' if self.hamming else 'without'))

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

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1]+1, 1, dtype=int), index=X.columns)
        A = np.hstack((np.ones((X.shape[0],1), dtype=int), X))
        # Iteration counter
        self.it = 0
        # Formulate master LP
        # Variables
        w = cvx.Variable(A.shape[1], nonneg=True)
        if self.CNF:
            if nN:
                xiN = cvx.Variable(nN, nonneg=True)
            if nU:
                xiU = cvx.Variable(nU, nonneg=True)
        else:
            xiO = cvx.Variable(nO, nonneg=True)
            if not self.hamming:
                xiN = cvx.Variable(nN)
                if nU:
                    xiU = cvx.Variable(nU)
        if not nU:
            self.gamma = 0
        # Objective function (no penalty on empty conjunction)
        lambdas = self.lambda0 + self.lambda1 * z.sum().values
        lambdas[0] = 0
        if not self.CNF and self.hamming:
            if nU:
                if nN:
                    obj = cvx.Minimize(cvx.sum(A[N,:] * w)/(nN*(1+self.gamma)) +\
                                       self.gamma * cvx.sum(A[U,:] * w)/(nU*(1+self.gamma)) +\
                                       lambdas * w)
                else:
                    obj = cvx.Minimize(cvx.sum(A[U,:] * w)/nU + lambdas * w)
            else:
                obj = cvx.Minimize(cvx.sum(A[N,:] * w)/ nN + lambdas * w)
        else:
            if nU:
                if nN:
                    obj = cvx.Minimize(cvx.sum(xiN)/(nN*(1+self.gamma)) +\
                                       self.gamma * cvx.sum(xiU)/(nU*(1+self.gamma)) +\
                                       lambdas * w)
                else:
                    obj = cvx.Minimize(cvx.sum(xiU) / nU + lambdas * w)
            else:
                obj = cvx.Minimize(cvx.sum(xiN) / nN + lambdas * w)
        # Constraints
        if self.CNF:
            if nN:
                constraints = [cvx.sum(A[O,:] * w) <= (1 - self.alpha) * nO,
                               xiN + A[N,:] * w >= 1]
            else:
                constraints = [cvx.sum(A[O,:] * w) <= (1 - self.alpha) * nO]

            if nU:
                constraints.append(xiU + A[U,:] * w >= 1)
        else:
            constraints = [cvx.sum(xiO) <= (1 - self.alpha) * nO,
                           xiO + A[O,:] * w >= 1]
            if not self.hamming:
                for (ii, i) in enumerate(N):
                    constraints.append(xiN[ii] >= cvx.max(w[A[i,:] > 0]))
                for (ii, i) in enumerate(U):
                    constraints.append(xiU[ii] >= cvx.max(w[A[i,:] > 0]))
        # Solve problem
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=self.verbose, solver=self.solver)

        # Extract dual variables
        r = np.zeros_like(y, dtype=float)
        if self.CNF:
            r[O] = constraints[0].dual_value
            if nN and nU:
                r[N] = -constraints[1].dual_value
                r[U] = -constraints[2].dual_value
            elif nU:
                r[U] = -constraints[1].dual_value
            else:
                r[N] = -constraints[1].dual_value
        else:
            r[O] = -constraints[1].dual_value
            if self.hamming:
                if nN and nU:
                    r[N] = 1. / (nN * (1+self.gamma))
                    r[U] = self.gamma / (nU * (1+self.gamma))
                elif nU:
                    r[U] = 1. / nU
                elif nN:
                    r[N] = 1. / nN
            else:
                r[N[xiN.value < self.eps]] = 1 / (nN * (1+self.gamma))
                if nU:
                    r[U[xiU.value < self.eps]] = self.gamma / (nU * (1+self.gamma))

        if not self.silent:
            self.logger.info('Initial solve completed')

        # Beam search for conjunctions with negative reduced cost
        if self.hamming:
            # Most negative reduced cost among current variables
            UB = np.dot(r, A) + lambdas
            #print('UB.min():', UB.min())
            UB = min(UB.min(), 0)
            v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                    K=self.K, UB=UB, eps=self.eps, B=self.B, D=self.D)
        else:
            v, zNew, Anew = beam_search_no_dup(r, X, self.lambda0, self.lambda1,
                    z, K=self.K, eps=self.eps, B=self.B, D=self.D)

        while (v < -self.eps).any() and (self.it < self.iterMax):
            # Negative reduced costs found
            self.it += 1

            if not self.silent:
                self.logger.info('Iteration: %d, Objective: %.4f' % (self.it, prob.value))

            # Add to existing conjunctions
            z = pd.concat([z, zNew], axis=1, ignore_index=True)
            A = np.concatenate((A, Anew), axis=1)

            # Reformulate master LP
            # Variables
            w = cvx.Variable(A.shape[1], nonneg=True)
            # Objective function
            lambdas = np.concatenate((lambdas, self.lambda0 + self.lambda1 * zNew.sum().values))
            if not self.CNF and self.hamming:
                if nU:
                    if nN:
                        obj = cvx.Minimize(cvx.sum(A[N,:] * w)/(nN*(1+self.gamma)) +\
                                       self.gamma * cvx.sum(A[U,:] * w)/(nU*(1+self.gamma)) +\
                                       lambdas * w)
                    else:
                        obj = cvx.Minimize(cvx.sum(A[U,:] * w)/nU + lambdas * w)
                else:
                    obj = cvx.Minimize(cvx.sum(A[N,:] * w)/ nN + lambdas * w)
            else:
                if nU and nN:
                    obj = cvx.Minimize(cvx.sum(xiN)/(nN*(1+self.gamma)) +\
                                       self.gamma * cvx.sum(xiU)/(nU*(1+self.gamma)) +\
                                       lambdas * w)
                elif nU:
                    obj = cvx.Minimize(cvx.sum(xiU) / nU + lambdas * w)
                else:
                    obj = cvx.Minimize(cvx.sum(xiN) / nN + lambdas * w)
            # Constraints
            if self.CNF:
                if nN:
                    constraints = [cvx.sum(A[O,:] * w) <= (1 - self.alpha) * nO,
                                   xiN + A[N,:] * w >= 1]
                else:
                    constraints = [cvx.sum(A[O,:] * w) <= (1 - self.alpha) * nO]

                if nU:
                    constraints.append(xiU + A[U,:] * w >= 1)
            else:
                constraints = [cvx.sum(xiO) <= (1 - self.alpha) * nO,
                               xiO + A[O,:] * w >= 1]
                if not self.hamming:
                    for (ii, i) in enumerate(N):
                        constraints.append(xiN[ii] >= cvx.max(w[A[i,:] > 0]))
                    for (ii, i) in enumerate(U):
                        constraints.append(xiU[ii] >= cvx.max(w[A[i,:] > 0]))
            # Solve problem
            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose=self.verbose, solver=self.solver)

            # Extract dual variables
            r = np.zeros_like(y, dtype=float)
            if self.CNF:
                r[O] = constraints[0].dual_value
                if nN and nU:
                    r[N] = -constraints[1].dual_value
                    r[U] = -constraints[2].dual_value
                elif nU:
                    r[U] = -constraints[1].dual_value
                else:
                    r[N] = -constraints[1].dual_value
            else:
                r[O] = -constraints[1].dual_value
                if self.hamming:
                    if nN and nU:
                        r[N] = 1. / (nN * (1+self.gamma))
                        r[U] = self.gamma / (nU * (1+self.gamma))
                    elif nU:
                        r[U] = 1. / nU
                    elif nN:
                        r[N] = 1. / nN
                else:
                    r[N[xiN.value < self.eps]] = 1 / (nN * (1+self.gamma))
                    if nU:
                        r[U[xiU.value < self.eps]] = self.gamma / (nU * (1+self.gamma))

            # Beam search for conjunctions with negative reduced cost
            if self.hamming:
                # Most negative reduced cost among current variables
                UB = np.dot(r, A) + lambdas
                #print('UB.min():', UB.min())
                UB = min(UB.min(), 0)
                v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                        K=self.K, UB=UB, eps=self.eps, B=self.B, D=self.D)
            else:
                v, zNew, Anew = beam_search_no_dup(r, X, self.lambda0, self.lambda1, z, K=self.K, eps=self.eps, D=self.D, B=self.B)

        # Save generated conjunctions and coefficients
        self.z = z
        w = w.value

        self.w_raw = w
        self.lp_obj_value = prob.value

        self.round_(X, y, scoring=self.rounding)

    def greedy_round_(self, X, y, xi=.5, use_lp=False, gamma=None):
        '''
        For DNF, this starts with no conjunctions, and adds them greedily
        based on a cost, which penalizes (any) inclusion of reference samples,
        and rewards (new) inclusion of positive samples, and goes until it
        covers at least alpha fraction of positive samples

        We do the following for CNF:
        + only consider rules that would adhere to limit on positive samples
        + add rules to cover (new) reference samples, while
        penalizing the coverage of positive samples
        '''

        A = self.compute_conjunctions(X)
        R = np.arange(0, A.shape[1]) # Remaining conjunctions
        U = []                       # Used conjunctions
        C = np.zeros(X.shape[0])     # Coverage indicator
        MAX_ITER = 1000
        i = 0
        gamma = self.gamma if gamma is None else gamma

        # Restrict conjunctions to those used by LP
        if use_lp:
            R = [R[i] for i in range(len(R)) if self.w_raw[i]>0]

        if self.CNF:
            while (i<MAX_ITER):
                assert (y == 0).sum() == 0, 'Neg samps not implemented for CNF'
                assert (y == -1).sum() > 0, 'No reference samples given' 

                # Frac of additional ref samples that each conjunction covers
                if A[(y == -1) & (C < 1), :].shape[0] == 0:
                    self.logger.info(
                        "Rounded rules cover all reference samples!")
                    break

                ref_new_cover = (A[(y == -1) & (C < 1),:][:,R] + 1e-8).mean(0)

                # Positive samples covered (for each conjunction)
                pos_cover = (A[(y == 1),:][:,R]).mean(0)

                # Regularization
                reg = self.lambda1 * self.z.values[:,R].sum(0)

                # Costs (for each conjunction)
                costs = xi*pos_cover - gamma*ref_new_cover + reg

                # Only consider feasible new rules, which maintain the
                # constraint that they cannot add too many pos samples
                # NOTE: This is a bit different than the actual constraint we
                # use in the LP relaxation, but is closer to what we actually
                # want
                feasible = (C[(y == 1), np.newaxis] + A[(y == 1),:][:, R]
                           ).mean(0) < 1 - self.alpha
                if feasible.sum() == 0:
                    break

                costs[~feasible] = np.inf

                r = np.argmin(costs)   # Find min-cost conjunction
                C = (C + A[:,R[r]])>0. # Update coverage
                U.append(R[r])
                R = np.array([R[i] for i in range(len(R)) if not i==r])

                i+=1

        else:
            while (i<MAX_ITER) and (C[y == 1].mean() < self.alpha):
                if (y==0).sum() > 0:
                    neg_cover = (A[(y == 0),:][:,R]).mean(0)
                else:
                    neg_cover = 0

                if (y==-1).sum() > 0:
                    # Fraction of reference samples that each conjunction covers
                    ref_cover = (A[(y == -1),:][:,R]).mean(0)
                else:
                    ref_cover = 0

                # Regularization (for each conjunction)
                reg = self.lambda1 * self.z.values[:,R].sum(0)

                # Positive samples newly covered (for each conjunction)
                pos_new_cover = (A[(y == 1) & (C < 1),:][:,R] + 1e-8).mean(0)

                # Costs (for each conjunction)
                costs = neg_cover + gamma*ref_cover + reg - xi*pos_new_cover

                r = np.argmin(costs)   # Find min-cost conjunction
                C = (C + A[:,R[r]])>0. # Update coverage
                U.append(R[r])
                R = np.array([R[i] for i in range(len(R)) if not i==r])

                i+=1


        # Zero out the rules and only take those which are used
        self.w = np.zeros(A.shape[1])
        self.w[U] = 1

    def round_(self, X, y, scoring='coverage', xi=None, use_lp=False, gamma=None, tol=0.01):

        """ Round based on scoring """
        if scoring == 'roc_auc':
            t_cand = np.unique(self.w_raw)
            best_auc = -1
            best_w = None
            for i in range(len(t_cand)):
                w = self.w_raw*(self.w_raw > t_cand[i])
                auc = roc_auc_score(y, self.predict_(X, w))
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            self.w = best_w

        elif scoring == 'greedy':
            self.greedy_round_(X, y, xi=xi, use_lp=use_lp, gamma=gamma)

        elif scoring == 'greedy_sweep':
            if xi is None:
                xi = np.logspace(np.log10(0.01), .5, 20)
            xis = np.array([xi]).ravel()
            best_xi = xis[0]
            if len(xis) > 1:
                best_xi = None
                best_auc = 0
                for xii in xis:
                    self.greedy_round_(X, y, xi=xii, use_lp=use_lp, gamma=gamma)
                    auc = roc_auc_score(y, self.predict(X))
                    if auc > best_auc - tol:
                        best_xi = xii
                    if auc > best_auc:
                        best_auc = auc
            self.greedy_round_(X, y, xi=best_xi, use_lp=use_lp, gamma=gamma)

        else:
            A = self.compute_conjunctions(X)
            w = self.w_raw
            O = np.where(y > 0)[0]
            nO = len(O)

            # Binarize coefficients
            # Candidates corresponding to all possible thresholds
            wCand = (w[:, np.newaxis] >= np.append(np.unique(w), 1.5)).astype(int)
            # Corresponding error variables
            if self.CNF:
                xiOCand = np.matmul(A[O,:], wCand)
            else:
                xiOCand = np.matmul(A[O,:], wCand) < 1
            # Candidates that satisfy overlap coverage constraint
            idxFeas = np.where(xiOCand.sum(axis=0) <= round((1 - self.alpha) * nO))[0]
            if self.CNF:
                # Choose the densest such candidate
                self.w = wCand[:, idxFeas[0]]
            else:
                # Choose the sparsest such candidate
                self.w = wCand[:, idxFeas[-1]]

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

        if not self.CNF:
            obj = np.sum(A[U,:].dot(w))/nU + lambdas.dot(w)
        else:
            obj = np.sum(np.maximum(1 - A[U, :].dot(w), 0))/nU + lambdas.dot(w)

        return obj

    def compute_conjunctions(self, X):
        """Compute conjunctions of features specified in self.z"""
        try:
            A = 1 - (np.dot(1 - X, self.z) > 0) # Changed matmul to dot, because failed on some machines
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        return A

    def predict_(self, X, w):
        """Predict whether points belong to overlap region"""
        # Compute conjunctions of features
        A = self.compute_conjunctions(X)
        # Predict labels
        if self.CNF:
            # Flip labels since model is actually a DNF for non-overlap
            return 1 - (np.dot(A, w) > 0)
        else:
            return (np.dot(A, w) > 0).astype(int)

    def predict(self, X):
        """Predict whether points belong to overlap region"""
        # Use helper function
        return self.predict_(X, self.w)
    
    def predict_rules(self, X):
        """Predict whether points belong to overlap region"""
        # Use helper function
        A = self.compute_conjunctions(X)
        
        if self.CNF:
            # Flip labels since model is actually a DNF for non-overlap
            # @TODO: Not sure if this is correct
            return 1 - (A*self.w > 0)
        else:
            return ((A*self.w) > 0).astype(int)

    def get_params(self, deep=False):
        """ Returns estimator parameters """
        # @TODO: Deep not implemented
        return dict([(k, getattr(self, k)) for k in self.valid_params])

    def set_params(self, **params):
        """ Sets estimator parameters """
        if not params:
            return self

        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)

        return self


if __name__ == '__main__':
    # Load iris-"plus" data for testing
    dirData = '../../Data/'
    datasets = pd.read_pickle(os.path.join(dirData, 'datasets.pkl'))
    ds = 'iris'
    d = datasets[ds]
    filePath = os.path.join(d['dirData'], d['fileName'] + '.csv')
    data = pd.read_csv(filePath, names=d['colNames'], header=d['rowHeader'], error_bad_lines=False)
    y = extract_target(data, **d)
    # Binarize all features including negations
    X = binarize_features(data, negations=True, **d)
