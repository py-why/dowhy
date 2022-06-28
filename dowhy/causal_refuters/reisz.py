import numpy as np
from scipy import stats
from econml.grf._base_grf import BaseGRF
from econml.utilities import cross_product
from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from dowhy.utils.util import get_numeric_features


def reisz_scoring(g, X):
    """
    Returns loss for a estimator model
    :param g: model to be evaluated
    :param X: data for testing the model

    :returns: floating point number that quantifies the model prediction
    Loss(g) = E[2*m(X,g) - g(X)**2]
    """
    loss = np.mean(2 * generate_moment_function(W=X,
                   g=g.predict) - g.predict(X) ** 2)
    return loss


def get_alpha_estimator(cv, X, max_degree, param_grid_dict):
    """
    Finds the best estimator for reisz representer (alpha_s )

    :param cv: training and testing data indices obtained afteer Kfolding the dataset
    :param X: training data for the model
    :param max_degree: degree of the polynomial function used to approximate alpha_s
    :param param_grid_dict: python dictionary with parameters to tune the ReiszRepresenter estimator

    :returns: estimator for alpha_s
    """
    if param_grid_dict is not None:
        estimator = GridSearchCVList([
            ReiszRepresenter(reisz_functions=create_polynomial_function(max_degree),
                             moment_function=generate_moment_function,
                             min_samples_leaf=10,
                             min_var_fraction_leaf=0.01,
                             max_depth=5)
        ],
            param_grid_list=[param_grid_dict],
            scoring=reisz_scoring,
            cv=cv,
            verbose=0,
            n_jobs=-1
        ).fit(X)
        return (estimator.best_estimator_)
    else:
        estimator = ReiszRepresenter(reisz_functions=create_polynomial_function(max_degree),
                                     moment_function=generate_moment_function,
                                     min_samples_leaf=10,
                                     min_var_fraction_leaf=0.01,
                                     max_depth=5).fit(X)
        return estimator


def get_generic_regressor(cv, X, Y, max_degree, estimator_list, estimator_param_list, numeric_features):
    """
    Finds the best estimator for reisz regression function (g_s)

    :param cv: training and testing data indices obtained afteer Kfolding the dataset
    :param X: regressors data for training the regression model
    :param Y: outcome data for training the regression model
    :param max_degree: degree of the polynomial function used to approximate the regression function
    :param estimator_list: list of estimator objects for finding the regression function
    :param estimator_param_list: list of dictionaries with parameters for tuning respective estimators in estimator_list
    :param numeric_features: list of indices of numeric features in the dataset

    :returns: estimator for Reisz Regression function
    """
    if estimator_list is not None:
        estimator = GridSearchCVList(estimator_list,
                                     estimator_param_list,
                                     cv=cv,
                                     scoring='neg_mean_squared_error',
                                     n_jobs=-1
                                     ).fit(X, Y)
        return estimator.best_estimator_
    else:
        estimator = GridSearchCVList([
            RandomForestRegressor(n_estimators=100, random_state=120),

            Pipeline([('scale', ColumnTransformer([('num', StandardScaler(), numeric_features)],
                                                  remainder='passthrough')),
                      ('lasso_model', Lasso())]),

            ReiszRegressor(regression_functions=create_polynomial_function(max_degree),
                           min_var_leaf_on_val=True,
                           min_impurity_decrease=1e-4,
                           max_samples=0.80,
                           inference=False,
                           subforest_size=1,
                           random_state=123),
            GradientBoostingRegressor()
        ],
            param_grid_list=[
                {'n_estimators': [50],
                 'max_depth': [3, 4, 5],
                 'min_samples_leaf': [10, 50]
                 },

                {'lasso_model__alpha': [0.01, 0.001, 1e-4, 1e-5, 1e-6]},

                {'regression_functions': [create_polynomial_function(2), create_polynomial_function(4)],
                 'min_samples_leaf': [10, 50],
                 'min_var_fraction_leaf': [0.01, 0.1],
                 'l2_regularizer': [1e-2, 1e-3]
                 },
                 {
                     'learning_rate' : [0.01, 0.001],
                     'n_estimators' : [50,200]
                 }
        ],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ).fit(X, Y)
        return estimator.best_estimator_


def generate_moment_function(W, g):
    """
    Generate and returns moment function
    m(W,g) = g(1,W) - g(0,W) for Average Causal Effect
    """
    shape = (W.shape[0], 1)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    non_treatment_data = W[:, 1:]
    data_0 = np.hstack([zeros, non_treatment_data])  # data with treatment = 1
    data_1 = np.hstack([ones, non_treatment_data])  # data with treatment = 0
    return g(data_1) - g(data_0)


def create_polynomial_function(max_degree):
    """
    Creates a list of polynomial functions

    :param max_degree: degree of the polynomial function to be created

    :returns: list of lambda functions
    """
    polynomial_function = []
    for degree in range(max_degree+1):
        def poly_term(x): return x[:, [0]] ** degree
        polynomial_function.append(poly_term)
    return polynomial_function


class ReiszRepresenter(BaseGRF):
    """
    Generalized Random Forest to estimate Reisz Representer (RR) 
    See: https://github.com/microsoft/EconML/blob/main/econml/grf/_base_grf.py
    :param reisz_functions: List of polynomial functions of n degree to approximate reisz representer created using create_polynomial_function
    :param moment_function: moment function m(W,g) whose expected value is used to calculate estimate
    :param l2_regularizer: l2 penalty while modeling (default = 1e-3)
    For tuning other parameters see https://econml.azurewebsites.net/_autosummary/econml.grf.CausalForest.html 
    """

    def __init__(self, *,
                 reisz_functions,
                 moment_function=generate_moment_function,
                 l2_regularizer=1e-3,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=0.45,
                 min_balancedness_tol=0.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):

        self.reisz_functions = reisz_functions
        self.moment_function = moment_function
        self.l2_regularizer = l2_regularizer
        self.num_reisz_functions = len(self.reisz_functions)

        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest,
                         inference=inference,
                         fit_intercept=False,
                         subforest_size=subforest_size,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

    def _get_alpha_and_pointJ(self, X, T, y):
        n_riesz_feats = len(self.reisz_functions)
        TX = np.hstack([T, X])
        riesz_feats = np.hstack([feat_fn(TX)
                                for feat_fn in self.reisz_functions])
        mfeats = np.hstack([self.moment_function(TX, feat_fn)
                            for feat_fn in self.reisz_functions])
        alpha = np.zeros((X.shape[0], n_riesz_feats))
        alpha[:, :n_riesz_feats] = mfeats
        riesz_cov_matrix = cross_product(riesz_feats, riesz_feats).reshape(
            (X.shape[0], n_riesz_feats, n_riesz_feats))
        penalty = self.l2_regularizer * np.eye(n_riesz_feats)
        penalty[0, 0] = 0
        pointJ = riesz_cov_matrix + penalty
        return alpha, pointJ.reshape((X.shape[0], -1))

    def _get_n_outputs_decomposition(self, X, T, y):
        n_relevant_outputs = len(self.reisz_functions)
        n_outputs = n_relevant_outputs
        return n_outputs, n_relevant_outputs

    def _translate(self, point, TX):
        riesz_feats = np.hstack([feat_fn(TX)
                                for feat_fn in self.reisz_functions])
        n_riesz_feats = riesz_feats.shape[1]
        riesz = np.sum(point[:, :n_riesz_feats] * riesz_feats, axis=1)
        return riesz

    def fit(self, X):
        return super().fit(X[:, 1:], X[:, [0]], X[:, [0]])

    def predict(self, X_test):
        point = super().predict(X_test[:, 1:])
        return self._translate(point, X_test)


class ReiszRegressor(BaseGRF):
    """
    A forest to estimate reisz regression function

    :param regression_functions: List of polynomial functions of n degree to approximate regression function created using create_polynomial_function
    :param l2_regularizer: l2 penalty while modeling (default = 1e-3)
    For tuning other parameters see https://econml.azurewebsites.net/_autosummary/econml.grf.CausalForest.html 

    """

    def __init__(self, *,
                 regression_functions,
                 l2_regularizer=1e-3,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):

        self.regression_functions = regression_functions
        self.num_reg_functions = len(self.regression_functions)
        self.l2_regularizer = l2_regularizer

        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest,
                         inference=inference,
                         fit_intercept=False,
                         subforest_size=subforest_size,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, y):
        return super().fit(X[:, 1:], X[:, [0]], y)

    def _get_alpha_and_pointJ(self, X, T, y):
        n_reg_feats = len(self.regression_functions)
        n_feats = n_reg_feats
        TX = np.hstack([T, X])
        reg_feats = np.hstack([feat_fn(TX)
                               for feat_fn in self.regression_functions])
        alpha = y.reshape(-1, 1) * reg_feats
        reg_cov_matrix = cross_product(reg_feats, reg_feats).reshape(
            (X.shape[0], n_reg_feats, n_reg_feats))
        penalty = self.l2_regularizer * np.eye(n_reg_feats)
        penalty[0, 0] = 0
        pointJ = reg_cov_matrix + penalty
        return alpha, pointJ.reshape((X.shape[0], -1))

    def _get_n_outputs_decomposition(self, X, T, y):
        n_relevant_outputs = len(self.regression_functions)
        n_outputs = n_relevant_outputs
        return n_outputs, n_relevant_outputs

    def _translate(self, point, TX):
        reg_feats = np.hstack([feat_fn(TX)
                               for feat_fn in self.regression_functions])
        reg = np.sum(point * reg_feats, axis=1)
        return reg

    def predict(self, X_test):
        point = super().predict(X_test[:, 1:])
        return self._translate(point, X_test)
