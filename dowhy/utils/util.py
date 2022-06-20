def get_numeric_features(X):
    """
    Finds the numeric feature columns in a dataset

    :param X: pandas dataframe

    returns: list of indices of numeric features
    """
    numeric_features_names = list(X.select_dtypes('number'))
    numeric_features = []
    for col_name in numeric_features_names:
        col_index = X.columns.get_loc(col_name)
        numeric_features.append(col_index)
    return numeric_features