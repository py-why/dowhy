# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson                #
# ----------------------------------------------#

import inspect
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from .. import support
from ..support import SupportEstimator
from ..overlap import SupportOverlapEstimator

class SVMSupportOverlapEstimator(SupportOverlapEstimator):
    """ Overlap estimator based on one-class SVM estimators of support """

    def __init__(self, alpha=.1, gamma_0='scale', gamma_1='scale', **kwargs):
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
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.kwargs = kwargs

        self.init_estimator_()

        self.valid_params = ['alpha', 'gamma_0', 'gamma_1']

    def init_estimator_(self):
        # Function handling input arguments
        constructor = SupportOverlapEstimator._get_estimator

        # Estimator for group 0
        self.S0 = support.SVMSupportEstimator(self.alpha, gamma=self.gamma_0,
                    **self.kwargs)

        # Estimator for group 1
        self.S1 = support.SVMSupportEstimator(self.alpha, gamma=self.gamma_1,
                    **self.kwargs)

    def grid_search(self, x, g, params, n_folds=5):

        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        g = g.values.ravel() if isinstance(g, pd.DataFrame) \
                     or isinstance(g, pd.Series) \
                     else g.ravel()

        ps = {'bandwidth': list(set([self.gamma_0, self.gamma_1]))}
        if 'gamma' in params:
            ps['bandwidth'] = params['gamma']
        if 'gamma_0' in params or 'gamma_1' in params:
            ps['bandwidth'] = params.get('gamma_0', []) \
                            + params.get('gamma_1', [])

        GS0 = GridSearchCV(KernelDensity(), ps, cv=n_folds, iid=False)
        GS0.fit(X.iloc[g==0])

        GS1 = GridSearchCV(KernelDensity(), ps, cv=n_folds, iid=False)
        GS1.fit(X.iloc[g==1])

        self.GS0 = GS0
        self.GS1 = GS1

        bp = {}
        if 'bandwidth' in GS0.best_params_:
            bp['gamma_0'] = GS0.best_params_['bandwidth']

        if 'bandwidth' in GS1.best_params_:
            bp['gamma_1'] = GS1.best_params_['bandwidth']

        self.set_params(**bp)

        return self

    def set_params(self, **params):
        """ Sets estimator parameters """
        if not params:
            return self

        reinit = False
        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)
        if reinit:
            self.init_estimator_()

        return self
