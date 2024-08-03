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
        encoder = OneHotEncoder(drop=drop, sparse_output=False)  # NB sparse renamed to sparse_output in sklearn 1.2+

        encoded_data = encoder.fit_transform(data_to_encode)

    else:  # Use existing encoder
        encoded_data = encoder.transform(data_to_encode)

    # Convert the encoded data to a DataFrame
    columns_encoded = encoder.get_feature_names_out(data_to_encode.columns)

    df_encoded = pd.DataFrame(encoded_data, columns=columns_encoded).reset_index(drop=True)  # drop index from original

    # Concatenate the encoded DataFrame with the original non-categorical columns
    df_result = pd.concat([df_columns_to_keep, df_encoded], axis=1)

    return df_result, encoder


class Encoders:
    """Categorical data One-Hot encoding helper object.

    Initializes a factory object which manages a set of sklearn.preprocessing.OneHotEncoder instances,
    although the `encode()` method can be overriden to replace these with your preferred encoder.

    Each Encoder instance is given a name to retrieve it in future, and is used to encode
    a different set of variables.
    """

    def __init__(self, drop_first=True):
        """Initializes an instance and calls `reset_encoders()`.

        :param drop_first: If true, will not encode the first category value with a bit in 1-hot encoding.
            It will be implicit instead, by the absence of any bit representing this value in the relevant columns.
            Set to False to include a bit for each value of every categorical variable.
        """
        self.drop_first = drop_first
        self.reset()

    def reset(self):
        """
        Removes any reference to data encoders, causing them to be re-created on next `encode()`.
        A separate encoder is used for each named set of variables.
        """
        self._encoders = {}

    def encode(self, data: pd.DataFrame, encoder_name: str):
        """
        Encodes categorical columns in the given data, returning a new dataframe containing
        all original data and the encoded columns. Numerical data is unchanged, categorical
        types are one-hot encoded. `encoder_name` identifies a specific encoder to be used
        if available, or created if not. The encoder can be reused in subsequent calls.

        :param data: Data to encode.
        :param encoder_name: The name for the encoder to be used.
        :returns: The encoded data.
        """
        existing_encoder = self._encoders.get(encoder_name)
        encoded_variables, encoder = one_hot_encode(
            data,
            drop_first=self.drop_first,
            encoder=existing_encoder,
        )

        # Remember encoder
        self._encoders[encoder_name] = encoder
        return encoded_variables
