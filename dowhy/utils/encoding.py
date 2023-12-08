import pandas as pd
from pandas.core.dtypes.common import is_list_like
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(data: pd.DataFrame, columns=None, drop_first: bool = False, encoder: OneHotEncoder = None):
    """
    Replaces pandas' get_dummies with an implementation of sklearn.preprocessing.OneHotEncoder.

    The purpose of replacement is to allow encoding of new data using the same encoder, which ensures that the resulting encodings are consistent.

    If encoder is None, a new instance of sklearn.preprocessing.OneHotEncoder will be created using `fit_transform()`. Otherwise, the existing encoder is used with `fit()`.

    For compatibility with get_dummies, the encoded data will be transformed into a DataFrame.

    In all cases, the return value will be the encoded data and the encoder object (even if passed in). If `data` contains other columns than the
    dummy-coded one(s), these will be prepended, unaltered, to the result.

    :param data: Data of which to get dummy indicators.
    :param columns: List-like structure containing specific columns to encode.
    :param drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level.
    :return: DataFrame, OneHotEncoder
    """

    # Determine columns being encoded
    if columns is None:
        dtypes_to_encode = ["object", "string", "category"]
        data_to_encode = data.select_dtypes(include=dtypes_to_encode)
    elif not is_list_like(columns):
        raise TypeError("Input must be a list-like for parameter `columns`")
    else:
        data_to_encode = data[columns]

    # If all columns are already numerical, there may be nothing to encode.
    # In this case, return original data.
    if len(data_to_encode.columns) == 0:
        return data, encoder  # Encoder may be None

    # Columns to keep in the result - not encoded.
    columns_to_keep = data.columns.difference(data_to_encode.columns)
    df_columns_to_keep = data[columns_to_keep].reset_index(drop=True)

    if encoder is None:  # Create new encoder
        drop = None
        if drop_first:
            drop = "first"
        encoder = OneHotEncoder(drop=drop, sparse=False)  # NB sparse renamed to sparse_output in sklearn 1.2+

        encoded_data = encoder.fit_transform(data_to_encode)

    else:  # Use existing encoder
        encoded_data = encoder.transform(data_to_encode)

    # Convert the encoded data to a DataFrame
    columns_encoded = encoder.get_feature_names_out(data_to_encode.columns)

    df_encoded = pd.DataFrame(encoded_data, columns=columns_encoded).reset_index(drop=True)  # drop index from original

    # Concatenate the encoded DataFrame with the original non-categorical columns
    df_result = pd.concat([df_columns_to_keep, df_encoded], axis=1)

    return df_result, encoder
