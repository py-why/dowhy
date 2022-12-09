# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson                #
# ----------------------------------------------#

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from ..overlap import OverlapEstimator

class KNNOverlapEstimator(OverlapEstimator):
    """ Overlap Estimator based on a k-Nearest Neighbors """

    def __init__(self, k=10, threshold=None, **kwargs):
        """ Initializes the estimator

        @args:
            k: The number of neighbors
            threshold: The fraction of each class necessary to deem overlapping
                       (Between 0 and .5). Set to 1/k if None
            kwargs: Additional keyword arguments passed to the KNN
        """
        self.k = k

        if threshold is None:
            threshold = 1./k

        self.threshold = threshold

        if 1/self.threshold > k:
            raise Exception('Resolution of threshold greater than largest \
                             possible for supplied k')

        if self.threshold > .5 or self.threshold < 0:
            raise Exception('Threshold out of range [0,.5]')

        # Initialize estimator
        self.M = KNeighborsClassifier(n_neighbors=self.k, **kwargs)

    def fit(self, x, g):
        """ Fits the overlap estimator to data in x

        @args:
            x: Empirical sample
            g: Group membership
        """
        self.M.fit(x, g.ravel())

        return self

    def predict(self, x):
        """ Predicts the overlap at a point x

        @args:
            x: Test point
        """

        return 1.0*(np.min(self.M.predict_proba(x),1)>=self.threshold)

    def score(self, x, o):
        """ Scores the fitted overlap estimator

        @args:
            x: Test points
            o: Overlap labels
        """
        return roc_auc_score(o.ravel(), self.predict(x))

    def score_base(self, x, g):
        """ Scores the k-NN estimator

        @args:
            x: Test points
            y: Test group membership
        """
        return roc_auc_score(g.ravel(), self.M.predict(x))


    def get_params(self, deep=False):
        """ Returns the parameters of the model
        """
        return {'k': self.k, 'threshold': self.threshold}

    def set_params(self, **params):
        """ Sets the parameters of the model
        """
        if not params:
            return self

        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'k' in params:
            self.k = params['k']

        return self

    def grid_search(self, x, g, params, n_folds=5):

        ps = {'n_neighbors': [self.k]}
        if 'k' in params:
            ps['n_neighbors'] = params['k']

        GS = GridSearchCV(KNeighborsClassifier(), ps, cv=n_folds, iid=False,
                scoring='roc_auc')
        GS.fit(x, g)

        bp = GS.best_params_
        if 'n_neighbors' in bp:
            bp['k'] = bp['n_neighbors']

        self.set_params(**bp)

        return self
