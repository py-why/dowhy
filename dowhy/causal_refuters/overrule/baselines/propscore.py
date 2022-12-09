# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson                #
# ----------------------------------------------#

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..overlap import OverlapEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d

class PropensityOverlapEstimator(OverlapEstimator):
    """ Overlap Estimator based on the propensity score """

    def __init__(self, threshold=.1, C=1., estimator=None, **kwargs):
        """ Initializes the estimator

        @args:
            threshold: The threshold of propensity score calipers
            kwargs: Additional keyword arguments passed to the propensity score
        """
        self.threshold = threshold
        self.C = C

        # Initialize estimator
        if estimator is not None:
            self.M = estimator
        else:
            self.M = LogisticRegression(C=C, solver='lbfgs', **kwargs)

    def fit(self, x, g):
        """ Fits the overlap estimator to data in x

        @args:
            x: Empirical sample
            g: Group membership
        """
        self.M.fit(x, g.ravel())
        self.Mr = self.M
        self.M = CalibratedClassifierCV(self.Mr, cv=2, method='isotonic')
        self.M.fit(x, g.ravel())

        return self

    def predict(self, x):
        """ Predicts the overlap at a point x

        @args:
            x: Test point
        """

        overlap = 1.0*(np.min(self.M.predict_proba(x),1) >= self.threshold)

        return overlap

    def score(self, x, y):
        """ Scores the fitted overlap estimator. Does not use the threshold.

        @args:
            x: Test points
            y: Overlap labels
        """
        return roc_auc_score(y, np.min(self.M.predict_proba(x),1))

    def get_params(self, deep=False):
        """ Returns the parameters of the model
        """
        return {'threshold': self.threshold}

    def set_params(self, **params):
        """ Sets the parameters of the model
        """
        if not params:
            return self

        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'C' in params:
            self.C = params['C']

        # Initialize estimator
        self.M = LogisticRegression(C=self.C, solver='lbfgs')
        print('WARNING: Not consistent with custom estimator')

        return self
