# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson                #
# ----------------------------------------------#

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from ..overlap import OverlapEstimator

class MarginalOverlapEstimator(OverlapEstimator):
    """ Overlap Estimator based on marginal histograms """

    def __init__(self, n_bins=20, **kwargs):
        """ Initializes the estimator

        @args:
            n_bins: The number histogram bins
            kwargs: Additional keyword arguments passed to the estimator
        """
        self.n_bins = n_bins

    def fit(self, x, g):
        """ Fits the overlap estimator to data in x

        @args:
            x: Empirical sample
            g: Group membership
        """

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(g, pd.DataFrame):
            g = g.values

        x0 = x[g.ravel()==0,:]
        x1 = x[g.ravel()==1,:]

        self.bins = []
        d = x.shape[1]
        for j in range(d):
            bins_j = np.percentile(x[:,j], np.linspace(0, 100, self.n_bins-1))
            bins_j = np.unique(bins_j)
            h0 = np.histogram(x0, np.hstack([-np.inf, bins_j, np.inf]))[0]
            h1 = np.histogram(x1, np.hstack([-np.inf, bins_j, np.inf]))[0]
            o_j = 1.0*((h0*h1)>0)
            self.bins.append((bins_j, o_j))

        return self

    def predict(self, x):
        """ Predicts the overlap at a point x

        @args:
            x: Test point
        """

        if isinstance(x, pd.DataFrame):
            x = x.values

        overlap = []
        for j in range(len(self.bins)):
            ids_j = np.digitize(x[:,j], self.bins[j][0])
            o_j = self.bins[j][1][ids_j]
            overlap.append(o_j)

        return np.prod(overlap, 0)

    def score(self, x, y):
        """ Scores the fitted overlap estimator

        @args:
            x: Test points
            y: Overlap labels
        """
        return roc_auc_score(y, self.predict(x))

    def get_params(self, deep=False):
        """ Returns the parameters of the model
        """
        return {'n_bins': self.n_bins}

    def set_params(self, **params):
        """ Sets the parameters of the model
        """
        if not params:
            return self

        if 'n_bins' in params:
            self.n_bins = params['n_bins']

        return self

class BoundingBoxOverlapEstimator(OverlapEstimator):
    """ Overlap Estimator based on marginal bounding boxes """

    def __init__(self, alpha=.1, **kwargs):
        """ Initializes the estimator

        @args:
            alpha: The mass of training data to cover
            kwargs: Additional keyword arguments passed to the estimator
        """
        self.alpha = alpha

    def fit(self, x, g):
        """ Fits the overlap estimator to data in x

        @args:
            x: Empirical sample
            g: Group membership
        """

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(g, pd.DataFrame):
            g = g.values

        x0 = x[g.ravel()==0,:]
        x1 = x[g.ravel()==1,:]

        lb = int(100.*self.alpha/2)
        ub = 100-lb

        self.bbs = []
        d = x.shape[1]
        for j in range(d):
            bbs0_j = np.percentile(x0[:,j], [lb, ub])
            bbs1_j = np.percentile(x1[:,j], [lb, ub])

            self.bbs.append((bbs0_j, bbs1_j))

        return self

    def predict(self, x):
        """ Predicts the overlap at a point x

        @args:
            x: Test point
        """

        if isinstance(x, pd.DataFrame):
            x = x.values

        overlap = []
        for j in range(len(self.bbs)):
            bb0, bb1 = (self.bbs[j][0], self.bbs[j][1])
            o_j = (x[:,j] >= bb0[0])*(x[:,j] <= bb0[1])\
                 *(x[:,j] >= bb1[0])*(x[:,j] <= bb1[1])
            overlap.append(o_j)

        return np.prod(overlap, 0)

    def score(self, x, y):
        """ Scores the fitted overlap estimator

        @args:
            x: Test points
            y: Overlap labels
        """
        return roc_auc_score(y, self.predict(x))

    def get_params(self, deep=False):
        """ Returns the parameters of the model
        """
        return {'alpha': self.alpha}

    def set_params(self, **params):
        """ Sets the parameters of the model
        """
        if not params:
            return self

        if 'alpha' in params:
            self.alpha = params['alpha']

        return self
