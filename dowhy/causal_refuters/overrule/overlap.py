# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson                #
# ----------------------------------------------#

import inspect
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from . import support
from .support import SupportEstimator

class OverlapEstimator(object):
    """ Interface for OverlapEstimators """
    def __init__(self):
        raise Exception('Abstract class.')

    def fit(self, x, g):
        raise Exception('Abstract class.')

    def predict(self, x):
        raise Exception('Abstract class.')

    def predict_proba(self, x):
        """ Predicts the probability of overlap at a point x
            @TODO: Is it bad form to have predict_proba here?

        @args:
            x: Test point
        """
        return np.array([1-self.predict(x), self.predict(x)]).T

    def score(self, x, o):
        return roc_auc_score(o, self.predict(x))

    def has_grid_search(self):
        return callable(getattr(self, 'grid_search', None))


class SupportOverlapEstimator(OverlapEstimator):
    """ Overlap estimator based on separate estimators of support """

    def __init__(self, support_estimator='svm', alpha=.1,
                 support_estimator_1=None, **kwargs):
        """ Initializes overlap estimator

        @args
            support_estimator: The class of estimator used for estimation of
                group-wise support in group 0 (and 1 if support_estimator_1 is
                None). Either a string in ['svm',...] or an implementation of
                support.SupportEstimator.
            alpha: The maximum mass excluded in support estimation
            support_estimator_1: Specific support estimator for class 1.
                Defaults to support_estimator if None is supplied.
            kwargs: Additional keyword arguments
        """

        self.alpha = alpha
        self.support_estimator = support_estimator
        self.support_estimator_1 = support_estimator_1
        self.kwargs = kwargs

        self.init_estimator_()

        self.valid_params = ['alpha', 'support_estimator',
            'support_estimator_1']

    def init_estimator_(self):
        # Function handling input arguments
        constructor = SupportOverlapEstimator._get_estimator

        # Estimator for group 0
        self.S0 = constructor(self.support_estimator,
            self.alpha, **self.kwargs)

        # Estiamtor for group 1
        if self.support_estimator_1 is None:
            self.S1 = constructor(self.support_estimator, self.alpha,
                        **self.kwargs)
        else:
            self.S1 = constructor(self.support_estimator_1, self.alpha,
                        **self.kwargs)

    @staticmethod
    def _get_estimator(support_estimator, alpha, **kwargs):
        """ Creates a support estimator based on supplied parameters

        @args
            support_estimator: The class of estimator used for estimation of
                group-wise support. Either a string in ['svm',...] or an implementation of support.SupportEstimator.
            alpha: The maximum mass excluded in support estimation
        """

        if inspect.isclass(support_estimator) and \
           issubclass(support_estimator, support.SupportEstimator):
            return support_estimator(alpha, **kwargs)
        elif isinstance(support_estimator, support.SupportEstimator):
            return support_estimator.copy()
        elif support_estimator == 'svm':
            params = kwargs
            params['gamma'] = params.get('gamma', 'scale')
            return support.SVMSupportEstimator(alpha, **params)
        else:
            raise Exception('Unknown overlap estimator type')

    def fit(self, x, g):
        """ Fit overlap estimator based on support estimation of p(x|g)

        @args
            x: Features
            g: Group indicator
        """
        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        g = g.values.ravel() if isinstance(g, pd.DataFrame) \
                     or isinstance(g, pd.Series) \
                     else g.ravel()

        x0 = X.loc[g == 0,:]
        x1 = X.loc[g == 1,:]

        self.S0.fit(x0)
        self.S1.fit(x1)

    def predict(self, x):
        """ Predict overlap at point x

        @args
            x: Features
        """
        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        s0 = self.S0.predict(X)
        s1 = self.S1.predict(X)
        o = s0*s1  # Indicator for overlap vs non-overlap

        return o

    def get_params(self, deep=False):
        """ Returns estimator parameters """
        # @TODO: Deep not implemented
        params = dict([(k, getattr(self, k)) for k in self.valid_params])

        return {**params, **self.kwargs}

    def set_params(self, **params):
        """ Sets estimator parameters """
        if not params:
            return self

        reinit = False
        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)
            elif k in self.M.valid_params:
                reinit = True
                self.kwargs[k] = v
        if reinit:
            self.init_estimator_()

        return self
