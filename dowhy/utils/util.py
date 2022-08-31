import numpy as np
from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_numeric_features(X):
    """
    Finds the numeric feature columns in a dataset

    :param X: pandas dataframe

    returns: list of indices of numeric features
    """
    numeric_features_names = list(X.select_dtypes("number"))
    numeric_features = []
    for col_name in numeric_features_names:
        col_index = X.columns.get_loc(col_name)
        numeric_features.append(col_index)
    return numeric_features


def get_generic_regressor(
    cv, X, Y, max_degree=3, estimator_list=None, estimator_param_list=None, numeric_features=None
):
    """
    Finds the best estimator for regression function (g_s)

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
        estimator = GridSearchCVList(
            estimator_list, estimator_param_list, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
        ).fit(X, Y)
        return estimator.best_estimator_
    else:
        estimator = GridSearchCVList(
            [
                RandomForestRegressor(n_estimators=100, random_state=120),
                Pipeline(
                    [
                        (
                            "scale",
                            ColumnTransformer([("num", StandardScaler(), numeric_features)], remainder="passthrough"),
                        ),
                        ("lasso_model", Lasso()),
                    ]
                ),
                GradientBoostingRegressor(),
            ],
            param_grid_list=[
                {"n_estimators": [50], "max_depth": [3, 4, 5], "min_samples_leaf": [10, 50]},
                {"lasso_model__alpha": [0.01, 0.001, 1e-4, 1e-5, 1e-6]},
                {"learning_rate": [0.01, 0.001], "n_estimators": [50, 200]},
            ],
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
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
    non_treatment_data = W[:, 1:]  # assume that treatment is one-dimensional.
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
    for degree in range(max_degree + 1):

        def poly_term(x):
            return x[:, [0]] ** degree

        polynomial_function.append(poly_term)
    return polynomial_function
