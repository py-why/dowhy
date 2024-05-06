"""
This module implements the estimators for causal change attribution in the following paper:
Quintas-Martinez, V., Bahadori, M. T., Santiago, E., Mu, J., Janzing, D., and Heckerman, D. 
Multiply-Robust Causal Change Attribution (ICML 2024)
https://arxiv.org/abs/2404.08839
"""

import warnings
from itertools import groupby
import numpy as np, pandas as pd, networkx as nx
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.weightstats import DescrStatsW
from dowhy.gcm.causal_models import ProbabilisticCausalModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dowhy.graph import DirectedGraph, node_connected_subgraph_view
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
from sklearn.model_selection import train_test_split

class ThetaC:
    """Implements three estimators (regression, re-weighting, MR) for causal change attribution."""

    def __init__(self, C, h_fn=lambda y: y, warn_th=1e-3):
        """
        Inputs:
        C = the change vector (a K+1 list of 0s and 1s).
        h_fn = the functional of interest. By default, the mean of y.
        """

        if any(x not in (0, 1) for x in C):
            raise ValueError(f"C must be a vector of 0s and 1s.")

        self.C = C
        self.h_fn = h_fn
        self.reg_dict = {}  # A dictionary to store the trained regressors
        self.cla_dict = {}  # A dictionary to store the trained classifiers
        self.calib_dict = {}  # A dictionary to store the trained calibrators
        self.alpha_dict = {} # A dictionary to store the fitted weights alpha_k (Theorem 2.4)
        self.warn_th = warn_th  # The threshold that generates warning about reweighting

    def _simplify_C(self, all_indep=False):
        """
        This function applies some simplifications to the change vector C,
        discussed in Appendix C of the paper.

        It creates:
        self.K_simpl: the number of groups after simplification (excluding the group that contains the outcome Y).
        self.C_simpl: the simplified change vector (a list of tuple. The first element in each tuple is a 0 or 1,
                      corresponding to the distribution we want to fix for the variables in that group. The second
                      element is a list containing the indices of the variables in that group).

        Inputs:
        all_indep = boolean, True if all explanatory variables are independent.
        """

        # When all variables are independent (Example C.4), simplify to a group of 0s and a group of 1s (regardless of order).
        if all_indep:
            unique = np.unique(self.C)
            self.C_simpl = sorted(
                [(c, [i for i in range(len(self.C)) if self.C[i] == c]) for c in unique],
                key=lambda a: np.max(a[1]),
            )

        # Otherwise, we just group the consecutive values (Remark C.1).
        else:
            self.C_simpl = [
                (c, list(inds)) for c, inds in groupby(range(len(self.C)), lambda i: self.C[i])
            ]

        self.K_simpl = len(self.C_simpl)-1

    def _train_reg(
        self,
        X_train,
        y_train,
        T_train,
        w_train=None,
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs=None,
        regressor_fit_kwargs=None,
    ):
        """
        This function trains the nested regression estimators, that will be stored in self.reg_dict.

        Inputs:
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_train = optional (n_train,) np.array with sample weights for the train data.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.
        """

        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs

        # Train gamma_K:
        ind = T_train == self.C_simpl[-1][0]  # Select sample C_{K+1} \in {0,1}
        var = [a for b in self.C_simpl[:-1] for a in b[1]]  # Select right variables
        self.reg_dict[self.K_simpl-1] = regressor(*regressor_args, **regressor_kwargs)
        if w_train is not None:
            self.reg_dict[self.K_simpl-1].fit(
                X_train[np.ix_(ind, var)], self.h_fn(y_train[ind]), sample_weight = w_train[ind],
                **regressor_fit_kwargs)
        else:
            self.reg_dict[self.K_simpl-1].fit(
                X_train[np.ix_(ind, var)], self.h_fn(y_train[ind]), **regressor_fit_kwargs)

        # Train gamma_k for k = K-1, K-2, ..., 1:
        for k in range(2, self.K_simpl+1):
            ind = T_train == self.C_simpl[-k][0]  # Select sample C_{k+1} \in {0,1}
            var_new = [a for b in self.C_simpl[:-k] for a in b[1]]  # Select right variables
            # Use the fitted values from previous regression
            new_y = self.reg_dict[self.K_simpl-k+1].predict(X_train[np.ix_(ind, var)])
            self.reg_dict[self.K_simpl-k] = regressor(*regressor_args, **regressor_kwargs)
            if w_train is not None:
                self.reg_dict[self.K_simpl-k].fit(X_train[np.ix_(ind, var_new)], new_y, sample_weight = w_train[ind],
                                                **regressor_fit_kwargs)
            else:
                self.reg_dict[self.K_simpl-k].fit(X_train[np.ix_(ind, var_new)], new_y,
                                                **regressor_fit_kwargs)
            var = var_new

    def _train_cla(
        self,
        X_train,
        T_train,
        X_eval,
        w_train=None,
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
    ):
        """
        This function trains the classification estimators for the weights, that will be stored in self.cla_dict.
        If calibrator is not None, it also calibrates the probabilities on a calibration set.

        Inputs:
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
                 Used only to give a warning about low overlap.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.

        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_calib = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.
        """

        classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        # Train classifiers that will go into alpha_k for k = 1, ..., K:
        for k in range(self.K_simpl):
            var = [a for b in self.C_simpl[:(k+1)] for a in b[1]]  # Select right variables
            self.cla_dict[k] = classifier(*classifier_args, **classifier_kwargs)
            if w_train is not None:
                self.cla_dict[k].fit(X_train[:, var], T_train, sample_weight = w_train, **classifier_fit_kwargs)
            else:
                self.cla_dict[k].fit(X_train[:, var], T_train, **classifier_fit_kwargs)

            # For the case where you want to calibrate on different data,
            # No need if classifier is CalibratedClassifierCv
            if calibrator is not None: 
                proba = self.cla_dict[k].predict_proba(X_calib[:, var])[:, [1]]
                self.calib_dict[k] = calibrator(*calibrator_args, **calibrator_kwargs)
                if w_train is not None:
                    self.calib_dict[k].fit(proba, T_calib, sample_weight = w_calib, **calibrator_fit_kwargs)
                else:
                    self.calib_dict[k].fit(proba, T_calib, **calibrator_fit_kwargs)

        var = [a for b in self.C_simpl[:-1] for a in b[1]]  # Select right variables
        p = self.cla_dict[self.K_simpl-1].predict_proba(X_eval[:, var])[:, 1]

        if np.min(p) < self.warn_th or np.max(p) > 1-self.warn_th:
            warnings.warn(
                f"min P(T = 1 | X) = {np.min(p) :.2f}, max P(T = 1 | X) = {np.max(p) :.2f}, indicating low overlap. \n"
                + "Consider increasing the regularization for the classificator or using method = 'regression'."
            )

    def _get_alphas(self, X_eval, T_eval, ratio, calibrator=None, crop=1e-3):
        """
        This helper function uses the classifiers (and, if appropriate, the probability calibrators)
        to compute the weights alpha_k (defined in Theorem 2.4 of the paper), 
        which are then stored in self.alpha_dict.

        Inputs:
        k = int from 0 to K_simpl-1.
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        ratio = n1/n0, unless the classifier has been trained with class weights.
        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
                     Used only to check if it is None.
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        alpha_k = (n0,) or (n1,) np.array of alpha_k weights for sample C_{k+1} \in {0,1}
        """

        for k in range(self.K_simpl):
            ind = T_eval == self.C_simpl[k+1][0]  # Select sample C_{k+1} \in {0,1}

            # k = 0 doesn't have parents, get the marginal RN derivative.
            if k == 0:
                var = self.C_simpl[0][1]  # Select right variables
                p = np.minimum(np.maximum(self.cla_dict[0].predict_proba(X_eval[np.ix_(ind, var)])[:, 1],
                                        crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[0].predict(p[:, np.newaxis]), crop), 1-crop)
                elif calibrator is not None and is_classifier(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[0].predict_proba(p[:, np.newaxis])[:, 1],
                                            crop), 1-crop)
                w = (1.0-p)/p * ratio if self.C_simpl[k+1][0] else p/(1.0-p) * 1/ratio

            # For k > 0 get the conditional RN derivative dividing the RN derivative for \bar{X}_j
            # by the RN derivative for \bar{X}_{j-1}.
            else:
                var_joint = [a for b in self.C_simpl[:(k+1)] for a in b[1]] # Variables up to k
                p_joint = np.minimum(np.maximum(self.cla_dict[k].predict_proba(X_eval[np.ix_(ind, var_joint)])[:, 1],
                                                crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p_joint = np.minimum(np.maximum(self.calib_dict[k].predict(p_joint[:, np.newaxis]), 
                                                    crop), 1-crop)
                elif calibrator is not None and is_classifier(calibrator):
                    p_joint = np.minimum(np.maximum(self.calib_dict[k].predict_proba(p_joint[:, np.newaxis])[:, 1],
                                                    crop), 1-crop)
                w_joint = (1-p_joint)/p_joint if self.C_simpl[k+1][0] else p_joint/(1-p_joint)

                var_cond = [a for b in self.C_simpl[:k] for a in b[1]]  # Variables up to k-1
                p_cond = np.minimum(np.maximum(self.cla_dict[k-1].predict_proba(X_eval[np.ix_(ind, var_cond)])[:, 1],
                                            crop), 1-crop)
                if calibrator is not None and is_regressor(calibrator):
                    p_cond = np.minimum(np.maximum(self.calib_dict[k-1].predict(p_cond[:, np.newaxis]),
                                                crop), 1-crop)
                if calibrator is not None and is_classifier(calibrator):
                    p = np.minimum(np.maximum(self.calib_dict[k-1].predict_proba(p_cond[:, np.newaxis])[:, 1],
                                            crop), 1-crop)
                w_cond = p_cond/(1-p_cond) if self.C_simpl[k+1][0] else (1-p_cond)/p_cond

                w = w_joint * w_cond

            self.alpha_dict[k] = w * self.alpha_dict[k-2] if k-2 in self.alpha_dict.keys() else w

        # alpha_k should integrate to 1. In small samples this might not be the case, so we standardize:
        self.alpha_dict[k] /= np.mean(self.alpha_dict[k])

    def est_scores(
        self,
        X_eval,
        y_eval,
        T_eval,
        X_train,
        y_train,
        T_train,
        w_eval=None,
        w_train=None,
        method="MR",
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs={},
        regressor_fit_kwargs={},
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
        all_indep=False,
        crop=1e-3,
    ):
        """
        This function computes the scores that are averaged to get each theta_hat.
        These are psi_hat in the notation of Section 2.5 of the paper.
        It is convenient to have a function that returns the scores,
        rather than just theta_hat, to compute things like bootstrapped standard errors.

        Inputs:
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        y_eval = (n_eval,) np.array with the Y data (outcome) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_eval = optional (n_eval,) np.array with sample weights for the evaluation set.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        method = One of 'regression', 're-weighting', 'MR'. By default, 'MR'.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.

        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.

        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_calib = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.

        all_indep = boolean, True if all explanatory variables are independent (used for self._simplify_C).
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        theta_scores = (n_eval,) np.array of scores, such that theta_hat = np.mean(theta_scores).
        """
        
        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs
        classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        if w_eval is None:
            n0, n1, n = np.sum(1-T_eval), np.sum(T_eval), T_eval.shape[0]
        else:
            n0, n1, n = np.sum(w_eval*(1-T_eval)), np.sum(w_eval*T_eval), np.sum(w_eval)

        if len(self.C) != X_train.shape[1]+1:
            raise ValueError(f"len(C) must be K+1={X_train.shape[1]+1}, not {len(self.C)}")

        self._simplify_C(all_indep=all_indep)

        if self.K_simpl > 0:
            if method == "regression":
                self._train_reg(
                    X_train,
                    y_train,
                    T_train,
                    w_train=w_train,
                    regressor=regressor,
                    regressor_args=regressor_args,
                    regressor_kwargs=regressor_kwargs,
                    regressor_fit_kwargs=regressor_fit_kwargs,
                )

                ind = T_eval == self.C_simpl[0][0]  # Select sample C_1 \in {0,1}
                var = self.C_simpl[0][1]  # Select right variables
                if self.C_simpl[0][0] == 1:
                    theta_scores = np.concatenate(
                        (
                            np.zeros_like(y_eval[T_eval==0]),
                            self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])*n/n1,
                        )
                    )
                else:
                    theta_scores = np.concatenate(
                        (
                            self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])*n/n0,
                            np.zeros_like(y_eval[T_eval==1]),
                        )
                    )

            elif method == "re-weighting":
                self._train_cla(
                    X_train,
                    T_train,
                    X_eval,
                    w_train=w_train,
                    classifier=classifier,
                    classifier_args=classifier_args,
                    classifier_kwargs=classifier_kwargs,
                    classifier_fit_kwargs=classifier_fit_kwargs,
                    calibrator=calibrator,
                    X_calib=X_calib,
                    T_calib=T_calib,
                    w_calib=w_calib,
                    calibrator_args=calibrator_args,
                    calibrator_kwargs=calibrator_kwargs,
                    calibrator_fit_kwargs=calibrator_fit_kwargs,
                )

                if 'class_weight' not in classifier_kwargs:
                    ratio = n1/n0
                elif classifier_kwargs['class_weight'] == 'balanced':
                    ratio = 1
                else:
                    ratio = n1/n0 * classifier_kwargs['class_weight'][0]/classifier_kwargs['class_weight'][1]

                self._get_alphas(
                    X_eval, 
                    T_eval, 
                    ratio,
                    calibrator=calibrator, 
                    crop=crop
                )
                
                ind = T_eval == self.C_simpl[-1][0]  # Select sample C_{K+1} \in {0,1}
                if self.C_simpl[-1][0] == 1:
                    theta_scores = np.concatenate(
                        (np.zeros_like(y_eval[T_eval==0]), self.alpha_dict[self.K_simpl-1] * self.h_fn(y_eval[ind])*n/n1)
                    )
                else:
                    theta_scores = np.concatenate(
                        (self.alpha_dict[self.K_simpl-1] * self.h_fn(y_eval[ind])*n/n0, np.zeros_like(y_eval[T_eval==1]))
                    )

            elif method == "MR":
                theta_scores_0 = np.zeros_like(y_eval[T_eval==0])
                theta_scores_1 = np.zeros_like(y_eval[T_eval==1])

                # Regression base estimate:
                self._train_reg(
                    X_train,
                    y_train,
                    T_train,
                    w_train=w_train,
                    regressor=regressor,
                    regressor_args=regressor_args,
                    regressor_kwargs=regressor_kwargs,
                    regressor_fit_kwargs=regressor_fit_kwargs,
                )

                ind = T_eval == self.C_simpl[0][0]  # Select sample C_1 \in {0,1}
                var = self.C_simpl[0][1]  # Select right variables
                if self.C_simpl[0][0] == 1:
                    theta_scores_1 += self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])
                else:
                    theta_scores_0 += self.reg_dict[0].predict(X_eval[np.ix_(ind, var)])

                # Debiasing terms up to K-1:
                self._train_cla(
                    X_train,
                    T_train,
                    X_eval,
                    w_train=w_train,
                    classifier=classifier,
                    classifier_args=classifier_args,
                    classifier_kwargs=classifier_kwargs,
                    classifier_fit_kwargs=classifier_fit_kwargs,
                    calibrator=calibrator,
                    X_calib=X_calib,
                    T_calib=T_calib,
                    w_calib=w_calib,
                    calibrator_args=calibrator_args,
                    calibrator_kwargs=calibrator_kwargs,
                    calibrator_fit_kwargs=calibrator_fit_kwargs,
                )

                if 'class_weight' not in classifier_kwargs:
                    ratio = n1/n0
                elif classifier_kwargs['class_weight'] == 'balanced':
                    ratio = 1
                else:
                    ratio = n1/n0 * classifier_kwargs['class_weight'][1]/classifier_kwargs['class_weight'][0]

                self._get_alphas(
                    X_eval, 
                    T_eval, 
                    ratio,
                    calibrator=calibrator, 
                    crop=crop
                )

                for k in range(self.K_simpl):
                    ind = T_eval == self.C_simpl[k+1][0]  # Select sample C_{k+1} \in {0,1}
                    var = [a for b in self.C_simpl[:(k+1)] for a in b[1]]  # Variables up to k
                    var_next = [a for b in self.C_simpl[:(k+2)] for a in b[1]]  # Variables up to k+1
                    if self.C_simpl[k+1][0] == 1:
                        if k < self.K_simpl-1:
                            theta_scores_1 += self.alpha_dict[k] * (
                                self.reg_dict[k+1].predict(X_eval[np.ix_(ind, var_next)])
                                - self.reg_dict[k].predict(X_eval[np.ix_(ind, var)])
                            )
                        else:
                            theta_scores_1 += self.alpha_dict[k] * (
                                self.h_fn(y_eval[ind])
                                - self.reg_dict[self.K_simpl-1].predict(X_eval[np.ix_(ind, var)])
                            )
                    else:
                        if k < self.K_simpl-1:
                            theta_scores_0 += self.alpha_dict[k] * (
                                self.reg_dict[k+1].predict(X_eval[np.ix_(ind, var_next)])
                                - self.reg_dict[k].predict(X_eval[np.ix_(ind, var)])
                            )
                        else:
                            theta_scores_0 += self.alpha_dict[k] * (
                                self.h_fn(y_eval[ind])
                                - self.reg_dict[self.K_simpl-1].predict(X_eval[np.ix_(ind, var)])
                            )

                theta_scores = np.concatenate((theta_scores_0*n/n0, theta_scores_1*n/n1))

            else:
                raise AttributeError(f'Method "{method}" Not Implemented')

        # When C = [1, 1, ..., 1] we can just take the sample mean of y_eval[T_eval == 1]
        elif self.C_simpl[0][0] == 1:
            theta_scores = np.concatenate((np.zeros_like(y_eval[T_eval==0]), self.h_fn(y_eval[T_eval == 1])*n/n1))

        # When C = [0, 0, ..., 0] we can just take the sample mean of y_eval[T_eval == 0]
        else:
            theta_scores = np.concatenate((self.h_fn(y_eval[T_eval == 0])*n/n0, np.zeros_like(y_eval[T_eval==1])))

        return theta_scores

    def est_theta(
        self,
        X_eval,
        y_eval,
        T_eval,
        X_train,
        y_train,
        T_train,
        w_eval=None,
        w_train=None,
        method="MR",  # One of 'regression', 're-weighting', 'MR',
        regressor=LinearRegression,
        regressor_args=(),
        regressor_kwargs=None,
        regressor_fit_kwargs=None,
        classifier=LogisticRegression,
        classifier_args=(),
        classifier_kwargs=None,
        classifier_fit_kwargs=None,
        calibrator=None,
        X_calib=None,
        T_calib=None,
        w_calib=None,
        calibrator_args=(),
        calibrator_kwargs=None,
        calibrator_fit_kwargs=None,
        all_indep=False,
        crop=1e-3,
    ):
        
        """
        This function computes the scores that are averaged to get each theta_hat,
        and then returns (theta_hat, std_error)

        Inputs:
        X_eval = (n_eval, K) np.array with the X data (explanatory variables) for the evaluation set.
        y_eval = (n_eval,) np.array with the Y data (outcome) for the evaluation set.
        T_eval = (n_eval,) np.array with the T data (sample indicator) for the evaluation set.
        X_train = (n_train, K) np.array with the X data (explanatory variables) for the training set.
        y_train = (n_train,) np.array with the Y data (outcome) for the training set.
        T_train = (n_train,) np.array with the T data (sample indicator) for the training set.
        w_eval = optional (n_eval,) np.array with sample weights for the evaluation set.
        w_train = optional (n_train,) np.array with sample weights for the training set.

        method = One of 'regression', 're-weighting', 'MR'. By default, 'MR'.

        regressor = the regression estimator: a class supporting .fit and .predict methods.
        regressor_args = a tuple of positional args for regressor.__init__.
        regressor_kwargs = a dictionary of keyword args for regressor.__init__.
        regressor_fit_kwargs = a dictionary of keyword args for regressor.fit.

        classifier = the classification estimator: a class supporting .fit and .predict_proba methods.
        classifier_args = a tuple of positional args for classifier.__init__.
        classifier_kwargs = a dictionary of keyword args for classifier.__init__.
        classifier_fit_kwargs = a dictionary of keyword args for classifier.fit.

        calibrator = Optional, a method for probability calibration on a calibration set.
                     This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                     a classifier (e.g. sklearn.LogisticRegression).
                     No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
        X_calib = (n_calib, K) np.array with the X data (explanatory variables) for the calibration set.
        T_calib = (n_calib,) np.array with the T data (sample indicator) for the calibration set.
        w_train = optional (n_calib,) np.array with sample weights for the calibration set.
        calibrator_args = a tuple of positional args for calibrator.__init__.
        calibrator_kwargs = a dictionary of keyword args for calibrator.__init__.
        calibrator_fit_kwargs = a dictionary of keyword args for calibrator.fit.

        all_indep = boolean, True if all explanatory variables are independent (used for self._simplify_C).
        crop = float, all predicted probabilities from the classifier will be cropped below at this lower bound,
               and above at 1-crop.

        Returns:
        theta_hat = the point estimate, np.mean(theta_scores) for the scores computed by self.est_scores.
        std_err = the standard error for theta_hat, sem(theta_scores) for the scores computed by self.est_scores.
        """

        regressor_kwargs = {} if regressor_kwargs is None else regressor_kwargs
        regressor_fit_kwargs = {} if regressor_fit_kwargs is None else regressor_fit_kwargs
        classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs
        classifier_fit_kwargs = {} if classifier_fit_kwargs is None else classifier_fit_kwargs
        calibrator_args = {} if calibrator_args is None else calibrator_args
        calibrator_fit_kwargs = {} if calibrator_fit_kwargs is None else calibrator_fit_kwargs

        theta_scores = self.est_scores(
            X_eval,
            y_eval,
            T_eval,
            X_train,
            y_train,
            T_train,
            w_eval=w_eval,
            w_train=w_train,
            method=method,  # One of 'regression', 're-weighting', 'MR',
            regressor=regressor,
            regressor_args=regressor_args,
            regressor_kwargs=regressor_kwargs,
            regressor_fit_kwargs=regressor_fit_kwargs,
            classifier=classifier,
            classifier_args=classifier_args,
            classifier_kwargs=classifier_kwargs,
            classifier_fit_kwargs=classifier_fit_kwargs,
            calibrator=calibrator,
            X_calib=X_calib,
            T_calib=T_calib,
            w_calib=w_calib,
            calibrator_args=calibrator_args,
            calibrator_kwargs=calibrator_kwargs,
            calibrator_fit_kwargs=calibrator_fit_kwargs,
            all_indep=all_indep,
            crop=crop,
        )

        if w_eval is not None:
            w_sort = np.concatenate((w_eval[T_eval==0], w_eval[T_eval==1])) # Order weights in same way as scores
        else:
            w_sort = np.ones(np.shape(T_eval))
        weighted_stats = DescrStatsW(theta_scores, weights=w_sort, ddof=0)

        return weighted_stats.mean, weighted_stats.std_mean

def distribution_change_robust(
    causal_model: ProbabilisticCausalModel,
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
    target_node: Any,
    sample_weight=None,
    xfit=True,
    xfit_folds=5,
    train_size=0.5, 
    calib_size=0.0,
    split_random_state = 0,
    method="MR",  # One of 'regression', 're-weighting', 'MR',
    regressor=LinearRegression,
    regressor_args=(),
    regressor_kwargs=None,
    regressor_fit_kwargs=None,
    classifier=LogisticRegression,
    classifier_args=(),
    classifier_kwargs=None,
    classifier_fit_kwargs=None,
    calibrator=None,
    calibrator_args=(),
    calibrator_kwargs=None,
    calibrator_fit_kwargs=None,
    all_indep=False,
    crop=1e-3,
    shapley_config: Optional[ShapleyConfig] = None,
):
    
    """
    This function computes the Shapley values for attribution of change in the mean of target_node
    to nodes upstream in the causal DAG, using the multiply-robust method from:
    Quintas-Martinez, V., Bahadori, M. T., Santiago, E., Mu, J., Janzing, D., and Heckerman, D. 
    Multiply-Robust Causal Change Attribution (ICML 2024)

    :param causal_model: Reference causal model.
    :param old_data: Joint samples from the 'old' distribution.
    :param new_data: Joint samples from the 'new' distribution.
    :param target_node: Target node of interest for attributing the marginal distribution change.
    :param sample_weight: Name of the sample weight variable, if using (optional).

    :param xfit: Whether to use cross-fitting (True) or sample splitting (False) to estimate the nuisance parameters.
    :param xfit_folds: Number of folds for cross-fitting if xfit = True.
    :param train_size: Share of observations in training set for nuisance parameters if xfit = False.
    :param calib_size: Share of observations in calibration set if calibrator is not None.
    :param split_random_state: Seed for sample splitting.

    :param method: One of 'regression', 're-weighting', 'MR'. By default, 'MR'.
    :param regressor: the regression estimator: a class supporting .fit and .predict methods.
    :param regressor_args: a tuple of positional args for regressor.__init__.
    :param regressor_kwargs: a dictionary of keyword args for regressor.__init__.
    :param regressor_fit_kwargs: a dictionary of keyword args for regressor.fit.

    :param classifier: the classification estimator: a class supporting .fit and .predict_proba methods.
    :param classifier_args: a tuple of positional args for classifier.__init__.
    :param classifier_kwargs: a dictionary of keyword args for classifier.__init__.
    :param classifier_fit_kwargs: a dictionary of keyword args for classifier.fit.

    :param calibrator: Optional, a method for probability calibration on a calibration set.
                    This could be a regressor (e.g. sklearn.isotonic.IsotonicRegression) or
                    a classifier (e.g. sklearn.LogisticRegression).
                    No need to do this if classifier is a sklearn.calibration.CalibratedClassifierCV learner.
    :param calibrator_args: a tuple of positional args for calibrator.__init__.
    :param calibrator_kwargs: a dictionary of keyword args for calibrator.__init__.
    :param calibrator_fit_kwargs: a dictionary of keyword args for calibrator.fit.

    :param all_indep: boolean, True if all explanatory variables are independent (used to simplify estimating equation).
    :param crop: float, all predicted probabilities from the classifier will be cropped below at this lower bound,
            and above at 1-crop.

    :param shapley_config: Configuration for the Shapley estimator.

    :return: A dictionary containing contribution of each upstream node.
    """

    causal_graph = causal_model.graph

    # Eliminate nodes that are not ancestry of the outcome:
    causal_graph = nx.DiGraph(node_connected_subgraph_view(causal_model.graph, target_node))

    # Sort by causal ancestry:
    sorted_var_names = list(nx.lexicographical_topological_sort(causal_graph))
    X_var_names = sorted_var_names[:-1]

    n0, n1 = old_data.shape[0], new_data.shape[0]

    X0 = old_data[X_var_names].values
    X1 = new_data[X_var_names].values
    y0 = old_data[target_node].values.flatten()
    y1 = new_data[target_node].values.flatten()
    X = np.concatenate((X0, X1))
    y = np.concatenate((y0, y1))
    T = np.concatenate((np.zeros(n0), np.ones(n1)))

    if not xfit:
        if sample_weight is None:
            X_train, X_eval, y_train, y_eval, T_train, T_eval = train_test_split(X, y, T, train_size = train_size, 
                                                                                 stratify = T, random_state = split_random_state)
            X_calib, T_calib = None, None
    
            if calibrator is not None:
                if calib_size == 0.0:
                    raise ValueError('For calibration, calib_size should be either positive and smaller than the number of samples or a float in the (0, 1) range.')
                    
                X_calib, X_train, _, y_train, T_calib, T_train = train_test_split(X_train, y_train, T_train, train_size = calib_size, 
                                                                                  stratify = T_train, random_state = split_random_state)
            
            w_eval, w_train, w_calib = None, None, None
    
        else:
            w = np.concatenate((old_data[sample_weight].values.flatten(), new_data[sample_weight].values.flatten()))
            
            X_train, X_eval, y_train, y_eval, T_train, T_eval, w_train, w_eval = train_test_split(X, y, T, w, train_size = train_size, 
                                                                                                  stratify = T, random_state = split_random_state)
            X_calib, T_calib, w_calib = None, None, None
    
            if calibrator is not None:
                if calib_size == 0.0:
                    raise ValueError('For calibration, calib_size should be either positive and smaller than the number of samples or a float in the (0, 1) range.')
                    
                X_calib, X_train, _, y_train, T_calib, T_train, w_calib, w_train = train_test_split(X_train, y_train, T_train, w_train, train_size = calib_size, 
                                                                                                    stratify = T_train, random_state = split_random_state)
    
        def set_func(C):
            return ThetaC(C, warn_th=0.0).est_theta(
                X_eval,
                y_eval,
                T_eval,
                X_train,
                y_train,
                T_train,
                w_eval=w_eval,
                w_train=w_train,
                method=method,  # One of 'regression', 're-weighting', 'MR',
                regressor=regressor,
                regressor_args=regressor_args,
                regressor_kwargs=regressor_kwargs,
                regressor_fit_kwargs=regressor_fit_kwargs,
                classifier=classifier,
                classifier_args=classifier_args,
                classifier_kwargs=classifier_kwargs,
                classifier_fit_kwargs=classifier_fit_kwargs,
                calibrator=calibrator,
                X_calib=X_calib,
                T_calib=T_calib,
                w_calib=w_calib,
                calibrator_args=calibrator_args,
                calibrator_kwargs=calibrator_kwargs,
                calibrator_fit_kwargs=calibrator_fit_kwargs,
                all_indep=all_indep,
                crop=crop
            )[0]
    
        attributions = estimate_shapley_values(set_func, len(sorted_var_names), shapley_config)

    if xfit:

        kf = KFold(n_splits = xfit_folds, shuffle = True, random_state = split_random_state)
        attributions = np.zeros(len(sorted_var_names))
        
        for (train_index, test_index) in kf.split(X):
            X_train, X_eval, y_train, y_eval, T_train, T_eval = X[train_index], X[test_index], y[train_index], y[test_index], T[train_index], T[test_index]
            w = np.concatenate((old_data[sample_weight].values.flatten(), new_data[sample_weight].values.flatten())) if sample_weight is not None else None
            w_train, w_eval = (w[train_index], w[test_index]) if sample_weight is not None else (None, None)
            
            X_calib, T_calib, w_calib = None, None, None
            
            if calibrator is not None:
                if calib_size == 0.0:
                    raise ValueError('For calibration, calib_size should be either positive and smaller than the number of samples or a float in the (0, 1) range.')

                if sample_weight is None:
                    X_calib, X_train, _, y_train, T_calib, T_train = train_test_split(X_train, y_train, T_train, train_size = calib_size, 
                                                                                  stratify = T_train, random_state = split_random_state)
                else:
                    X_calib, X_train, _, y_train, T_calib, T_train, w_calib, w_train = train_test_split(X_train, y_train, T_train, w_train, train_size = calib_size, 
                                                                                                        stratify = T_train, random_state = split_random_state)

            def set_func(C):
                return ThetaC(C, warn_th=0.0).est_theta(
                    X_eval,
                    y_eval,
                    T_eval,
                    X_train,
                    y_train,
                    T_train,
                    w_eval=w_eval,
                    w_train=w_train,
                    method=method,  # One of 'regression', 're-weighting', 'MR',
                    regressor=regressor,
                    regressor_args=regressor_args,
                    regressor_kwargs=regressor_kwargs,
                    regressor_fit_kwargs=regressor_fit_kwargs,
                    classifier=classifier,
                    classifier_args=classifier_args,
                    classifier_kwargs=classifier_kwargs,
                    classifier_fit_kwargs=classifier_fit_kwargs,
                    calibrator=calibrator,
                    X_calib=X_calib,
                    T_calib=T_calib,
                    w_calib=w_calib,
                    calibrator_args=calibrator_args,
                    calibrator_kwargs=calibrator_kwargs,
                    calibrator_fit_kwargs=calibrator_fit_kwargs,
                    all_indep=all_indep,
                    crop=crop
                )[0]

            attributions += estimate_shapley_values(set_func, len(sorted_var_names), shapley_config)/xfit_folds
    
    
    return {x: attributions[i] for i, x in enumerate(sorted_var_names)}
