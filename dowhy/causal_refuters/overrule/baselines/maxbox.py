# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Tian Gao, Fredrik D. Johansson      #
# ----------------------------------------------#

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import pandas as pd

# Import r packages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

# Add parent to path for relative import
sys.path.append('..')

# Import OverRule packages
from ..ruleset import RulesetEstimator

class MaxBoxRulesetEstimator(RulesetEstimator):
    """
    Max-box overlap estimation
    """

    def __init__(self, cutoff=10, max_iter=10000, source_path='./maxbox.R'):
        """Initializes the estimator

        @args:
            cutoff: At which point should the branch and bound switch from a depth
                first to best first strategy. Can be helpful to make this nonzero
                in order to get an initial feasible solution
            max_iter: maximum number of iterations. If this is exceeded, we instead
                use the best solution found up until that point (even though it may
                have suboptimal cardinality)
            source_path: Relative path of R-script
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), source_path)
        self.cutoff = cutoff
        self.max_iter = max_iter
        self.r = ro.r
        self.r.source(path)

    def fit(self, x, o):
        """ Fit the overlap region based on a sample x and an indicator for
        overlap, denoted o

        @args:
            x: Samples
            o: Binary indicator for overlap, where 1 indicates overlap
        """

        # Construct features dataframe
        x = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        bounds, cardinality, index = self.r.maxbox(x.values, 1-o, self.cutoff, self.max_iter)

        self.bounds = np.array(bounds)

        self.lb = pd.DataFrame(self.bounds[:,0:1].T, columns=x.columns.values).iloc[0]
        self.ub = pd.DataFrame(self.bounds[:,1:2].T, columns=x.columns.values).iloc[0]

        self.bounds = bounds
        self.cardinality = np.array(cardinality)
        self.index = np.array(index)

        return self

    def predict(self, x):
        """ Predict whether or not X lies in the overlap region (1 = True)

        @args:
            x: Test point
        """

        # Construct features dataframe
        x = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        return np.product(self.predict_rules(x), axis=1)

    def predict_rules(self, x):
        """ Compute rules satisfied by x

        @args:
            x: Test point
        """

        # Construct features dataframe
        x = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        lb = (x >= self.lb)
        ub = (x <= self.ub)
        d = x.shape[1]
        R = pd.concat([lb, ub], axis=1)

        xs = x.columns.values
        lbs = ['%s >= %.3f..' % (xs[i], self.lb[i]) for i in range(x.shape[1])]
        ubs = ['%s <= %.3f..' % (xs[i], self.ub[i]) for i in range(x.shape[1])]
        R.columns = lbs + ubs
        return R

    def rules(self, as_str=False, transform=None, fmt=None, labels={}):
        """ Returns rules learned by the estimator

        @args:
            as_str: Returns a string if True, otherwise a dictionary
            transform: A function that takes key-value pairs for rules and
                thresholds and transforms the value. Used to re-scale
                standardized data for example
            fmt: Formatting string for float values
            labels: Renaming of the columns
        """

        def t_(k, v):
            return v if transform is None else transform(k, v)

        xs = self.lb.index.values
        ls = [labels.get(s,s) for s in xs]

        lb = self.lb.copy()
        ub = self.ub.copy()
        for k in xs:
            lb[k] = t_(k, lb[k])
            ub[k] = t_(k, ub[k])


        R = pd.DataFrame(np.vstack([lb.values, ub.values]), columns=ls, index=['>=', '<=']).T

        # @TODO: Add formatting. Currently doesn't work

        if as_str:
            return str(R)
            #if fmt is not None:
            #    return str(R.apply(lambda x : fmt % x))
            #else:
            #    return str(R)

        return R
