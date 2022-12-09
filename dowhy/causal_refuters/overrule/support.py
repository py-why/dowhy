# ------------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets    #
# @Authors: Fredrik D. Johansson, Michael Oberst  #
# ------------------------------------------------#

import numpy as np
from sklearn import svm


class SupportEstimator():
    """ Estimator of the support of multivariate densities"""
    def __init__(self, alpha=.1, **kwargs):
        self.alpha = alpha

    def fit(self, x):
        """ Fit the support of densities based on a sample x

        @args:
            x: Sample from a density p
        """
        pass

    def predict(self, x):
        """ Predict the support status at a point x (or a set of points)

        @args:
            x: Test point
        """
        pass

    def copy(self):
        """ Returns a new model with the same parameters
        """
        raise Exception('Not implemented.')

    def score(self, x):
        """ Scores the fitted support region based on @TODO: What?

        @args:
            x: Test points
        """
        pass


class SVMSupportEstimator(SupportEstimator):
    """ Support Estimator based on the One-Class SVM """

    def __init__(self, alpha=.1, **kwargs):
        """ Initializes the estimator

        @args:
            alpha: The largest fraction of samples not covered by the support
            kernel: The kernel used in the One-Class SVM
            gamma: The kernel coefficient
            kwargs: Additional keyword arguments passed to the OneClassSVM
        """
        self.alpha = alpha
        self.kwargs = kwargs

        # Initialize estimator
        self.M = svm.OneClassSVM(nu=self.alpha, **kwargs)

        # Add decision_function as a member function
        self.decision_function = self.M.decision_function

    def copy(self):
        """ Returns a new model with the same parameters
        """
        return SVMSupportEstimator(self.alpha, **self.kwargs)

    def fit(self, x):
        """ Fits the Support estimator to data in x

        @args:
            x: Empirical sample
        """
        self.M.fit(x)
        return self

    def predict(self, x):
        """ Predicts the support value at a point x

        @args:
            x: Test point
        """
        return .5*(self.M.predict(x) + 1)

    def score(self, x):
        """ Scores the fitted support region based on @TODO: What?

        @args:
            x: Test points
        """
        return (1-self.predict(x).mean())-self.alpha
