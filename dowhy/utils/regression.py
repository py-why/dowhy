import numpy as np


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
