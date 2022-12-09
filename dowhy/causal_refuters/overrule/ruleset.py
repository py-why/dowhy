# -----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets   #
# @Authors: Fredrik D. Johansson, Michael Oberst #
# -----------------------------------------------#

import numpy as np
import pandas as pd

# For Decision Tree Classifier
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_auc_score

# For BCS
from .BCS.overlap_boolean_rule import OverlapBooleanRule
from .BCS.load_process_data_BCS import extract_target, FeatureBinarizer

# Overrule imports
from .utils import sampleUnif, sample_reference, rule_str


class RulesetEstimator(object):
    """
    RulesetEstimator

    Binary classification of overlap / non-overlap with a binary classifier,
    along with corresponding rules, which may be in different formats based
    on the underlying classifier used
    """
    def __init__(self, n_ref_multiplier=1.):
        pass

    def fit(self, x, s):
        """ Fit the overlap region based on a sample x and an indicator for
        overlap, denoted s. If it is necessary to generate a reference measure,
        then do so here

        @args:
            x: Samples
            s: Binary indicator for overlap, where 1 indicates overlap
        """
        pass

    def predict(self, x):
        """ Predict whether or not X lies in the overlap region (1 = True)

        @args:
            x: Test point
        """
        pass

    def predict_proba(self, x):
        """ Predicts the probability of overlap at a point x
            @TODO: Is it bad form to have predict_proba here?

        @args:
            x: Test point
        """
        return np.array([1-self.predict(x), self.predict(x)]).T

    def describe(self, outpath):
        """ Returns a description of the model

        @args:
            outpath: A file to write the desciption to
        """
        pass

    def set_params(self, **params):
        pass

    def get_params(self):
        pass


class DTRulesetEstimator(RulesetEstimator):
    """Ruleset Estimator based on Decision Trees"""

    def __init__(self, n_ref_multiplier=1., **kwargs):
        """Initializes the estimator

        @args:
            kwargs: Keyword arguments for the DecisionTreeClassifier
        """
        self.n_ref_multiplier = n_ref_multiplier
        self.kwargs = kwargs

        # Bookkeeping
        self.refSamples = None
        self.overlapSamples = None

        # Initialize estimator
        self.init_estimator_()

        # Valid parameters
        self.valid_params = ['n_ref_multiplier']

    def init_estimator_(self):
        """ Init rule set estimator """
        self.M = DecisionTreeClassifier(**self.kwargs)

    def fit(self, x, o):
        """ Fit the overlap region based on a sample x and an indicator for
        overlap, denoted s

        @args:
            x: Samples
            o: Binary indicator for overlap, where 1 indicates overlap
        """
        dim = x.shape[1]
        nRef = int(x.shape[0] * dim * self.n_ref_multiplier)

        # Convert to dataframe if not
        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        # Format labels
        o = o.values.ravel() if (isinstance(o, pd.DataFrame)
            or isinstance(o, pd.Series)) else o.ravel()

        # Sample from reference measure and construct features
        # @TODO: Should not be uniform noise if binary features
        self.refSamples = sample_reference(X, n=nRef)

        # Add reference samples
        data = pd.concat([X, self.refSamples], 0)
        o = np.hstack([o, -np.ones(nRef)])

        # Fit model
        self.M.fit(data, o)

    def predict(self, x):
        """ Predict whether or not X lies in the overlap region (1 = True)

        @args:
            x: Test point
        """
        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        return self.M.predict(X)

    def describe(self, outpath):
        """ Returns a description of the model """
        dot_data = export_graphviz(self.M,
                                   filled=True, rounded=True,
                                   out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(outpath)

    def get_params(self, deep=False):
        """ Returns estimator parameters """
        # @TODO: Deep not implemented
        params = dict([(k, getattr(self, k)) for k in self.valid_params])

        if deep:
            return {**params, **self.M.get_params(deep=True)}
        else:
            return params

    def set_params(self, **params):
        """ Sets estimator parameters """
        if not params:
            return self

        reinit = False
        for k, v in params.items():
            if k in self.valid_params:
                setattr(self, k, v)
            elif k in self.M.get_params(deep=True):
                reinit = True
                self.kwargs[k] = v
        if reinit:
            self.init_estimator_()

        return self


class BCSRulesetEstimator(RulesetEstimator):
    """Ruleset Estimator based on ./BCS"""

    def __init__(self, lambda0=0., lambda1=0., cat_cols=[],
                 n_ref_multiplier=1., negations=True, num_thresh=9,
                 seed=None, ref_range=None, thresh_override=None, **kwargs):
        """Initializes the estimator

        @args:
            lambda0: Regularization of #rules
            lambda1: Regularization of #literals
            cat_cols: Set of categorical columns
            n_ref_multiplier: Reference sample count multiplier
            negations: Whether to use negations of literals
            num_thresh: Number of bins for continuous variables
            seed: Random seed for reference samples
            ref_range: Manual override of the range for reference samples, given as a
                       dictionary of {column_name: 
                                       {'is_binary': true/false,
                                        'min': min
                                        'max': max}
                                      }
            thresh_override: Manual override of the thresholds for continuous
                features, given as a dictionary like this, will only be applied
                to continuous features with more than num_thresh unique values
                    {column_name: np.linspace(0, 100, 10)}
            kwargs: Keyword arguments for the OverlapBooleanRule
                    (see ./BCS/overlap_boolean_rule.py for description of arguments)
        """

        # Parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.cat_cols = cat_cols
        self.n_ref_multiplier = n_ref_multiplier
        self.negations = negations
        self.num_thresh = num_thresh
        self.seed = seed
        self.ref_range = ref_range
        self.thresh_override = thresh_override

        # @TODO: something not right if these are set for the constructor
        # using partial() and then passed to GridSearchCV. not passed on.
        self.kwargs = kwargs

        # Bookkeeping
        self.refSamples = None
        self.overlapSamples = None

        # Initialize estimators
        self.init_estimator_()

        # @TODO: Make class variable?
        self.valid_params = ['lambda0', 'lambda1', 'cat_cols',
            'n_ref_multiplier', 'negations', 'num_thresh', 'seed']

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in self.kwargs.keys():
            state['kwargs']['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def init_estimator_(self):

        """ Init rule set estimator and binarizer """
        self.M = OverlapBooleanRule(lambda0=self.lambda0,
                    lambda1=self.lambda1, **self.kwargs)

        self.FeatureBinarizer = FeatureBinarizer(negations=self.negations,
                    colCateg=self.cat_cols, numThresh=self.num_thresh,
                    threshOverride=self.thresh_override)

    def fit(self, x, o):
        """ Fit the overlap region based on a sample x and an indicator for
        overlap, denoted o

        @args:
            x: Samples
            o: Binary indicator for overlap, where 1 indicates overlap
        """

        n = x.shape[0]
        dim = x.shape[1]
        nRef = int(n * dim * self.n_ref_multiplier)

        # Convert to dataframe if not
        X = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        # Format labels
        o = o.values.ravel() if (isinstance(o, pd.DataFrame)
            or isinstance(o, pd.Series)) else o.ravel()

        # Sample from reference measure and construct features
        self.refSamples = sample_reference(X, n=nRef, seed=self.seed,
                                           ref_range=self.ref_range)

        # Add reference samples
        data = pd.concat([X, self.refSamples], axis=0, sort=False)
        o = np.hstack([o, -np.ones(nRef)])

        # Binarize features (fit to data only)
        self.FeatureBinarizer.fit(data.iloc[:n])
        X = self.FeatureBinarizer.transform(data)

        # Fit estimator
        self.M.fit(X, o)

        # Store reference volume
        if nRef > 0:
            self.relative_volume = self.predict(self.refSamples).mean()

        return self

    def predict(self, x):
        """ Predict whether or not X lies in the overlap region (1 = True)

        @args:
            x: Test point
        """
        # Construct features dataframe
        data = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        X = self.FeatureBinarizer.transform(data).fillna(0)

        preds = self.M.predict(X)

        return preds

    def predict_rules(self, x):
        """ Predict rules activated by x

        @args:
            x: Test point
        """
        # Construct features dataframe
        data = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        X = self.FeatureBinarizer.transform(data).fillna(0)

        return self.M.predict_rules(X)

    def get_objective_value_(self, x, o):
        # Construct features dataframe
        data = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        X = self.FeatureBinarizer.transform(data).fillna(0)

        return self.M.get_objective_value(X, o)

    def round_(self, x, o, scoring='roc_auc', **kwargs):
        """ Round rule set """
        # Construct features dataframe
        data = x if isinstance(x, pd.DataFrame) \
            else pd.DataFrame(dict([('x%d' % i, x[:,i]) for i in range(x.shape[1])]))

        X = self.FeatureBinarizer.transform(data).fillna(0)

        self.M.round_(X, o, scoring=scoring, **kwargs)

    def rules(self, as_str=False, transform=None, fmt='%.3f', labels={}):
        """ Returns rules learned by the estimator

        @args:
            as_str: Returns a string if True, otherwise a dictionary
            transform: A function that takes key-value pairs for rules and
                thresholds and transforms the value. Used to re-scale
                standardized data for example
            fmt: Formatting string for float values
        """
        w, z = (self.M.w, self.M.z)
        w_sel = np.where(w)[0]


        def t_(k, v):
            return v if transform is None else transform(k, v)

        C = []
        for j in w_sel:
            index_j = z[z[j]==1][j].index
            f = index_j.get_level_values(0).values
            o = index_j.get_level_values(1).values
            v = index_j.get_level_values(2).values

            l = [labels.get(a, a) for a in f]

            dis_j = [(l[i], o[i], t_(f[i], v[i])) for i in range(len(f))]
            C.append(dis_j)

        if as_str:
            return rule_str(C, fmt=fmt)
        else:
            return C

    def describe(self, outpath):
        """ Returns a description of the model """
        with open(outpath, 'w') as f:
            f.write('Conjunctions:\n')
            f.write(self.M.z)
            f.write('\n\nCoefficients:\n')
            f.write(self.M.w)

    def complexity(self):
        """ Returns number of rules and number of atoms """
        rules_o = self.rules()

        n_rules_o = len(rules_o)
        n_atoms_o = np.sum([len(r) for r in rules_o])

        return n_rules_o, n_atoms_o

    def score(self, x, y):
        """ Evaluates the fitted models. If a label y is supplied, the score
            measures the accuracy of the ruleset estimation
        @args
            x: Test point
            y: Label indicating the label at x
        """
        return roc_auc_score(y, self.predict(x))

    def get_params(self, deep=False):
        """ Returns estimator parameters """
        params = dict([(k, getattr(self, k)) for k in self.valid_params])

        if deep:
            return {**params, **self.M.get_params(deep=True)}
        else:
            return params

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
