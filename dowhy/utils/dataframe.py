# Required imports for a robust implementation
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
from typing import Any, Dict, Iterable, List, Literal, Tuple


# Type aliases for objects related to the dataframe interchange protocol.
DataFrameObject = Any
InterchangeObject = Any
InterchangeColumn = Any

# Mapping from the protocol's DType enumeration to NumPy's dtype objects
# See: https://data-apis.org/dataframe-protocol/latest/API.html#dtype
_INTERCHANGE_TO_NUMPY_DTYPE = {
    1: np.int8,  # INT8
    2: np.int16,  # INT16
    3: np.int32,  # INT32
    4: np.int64,  # INT64
    5: np.uint8,  # UINT8
    6: np.uint16,  # UINT16
    7: np.uint32,  # UINT32
    8: np.uint64,  # UINT64
    9: np.float16,  # FLOAT16
    10: np.float32,  # FLOAT32
    11: np.float64,  # FLOAT64
    20: np.bool_,  # BOOL
    21: np.object_,  # STRING (UTF-8) -> becomes object array in NumPy
}


class DataFrameWrapper:
    """
    A robust, type-safe wrapper for dataframe objects that implement the
    Python dataframe interchange protocol (`__dataframe__`).

    This class provides a standardized API to access and manipulate dataframe
    data, abstracting away the specific implementation (e.g., pandas, polars).
    """

    def __init__(self, df: DataFrameObject):
        if not hasattr(df, "__dataframe__"):
            raise TypeError(
                f"The provided object of type '{type(df).__name__}' does not support "
                "the dataframe interchange protocol."
            )

        self._df_original: DataFrameObject = df
        self._interchange_obj: InterchangeObject = df.__dataframe__(allow_copy=False)
        self._original_type: type = type(df)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the (number of rows, number of columns) of the dataframe."""
        return (self._interchange_obj.num_rows(), self._interchange_obj.num_columns())

    @property
    def columns(self) -> List[str]:
        """Returns the names of the columns in the dataframe."""
        return self._interchange_obj.column_names()

    @property
    def dtypes(self) -> Dict[str, Any]:
        """Returns a dictionary mapping column names to their NumPy dtype."""
        return {
            name: _INTERCHANGE_TO_NUMPY_DTYPE.get(col.dtype[0], np.object_)
            for name, col in self._interchange_obj.get_columns()
        }

    def __getitem__(self, value: str | List[str]) -> DataFrameWrapper | ColumnWrapper:
        """Provides dict-like access to columns."""
        if isinstance(value, str):
            return self.get_column(value)
        elif isinstance(value, list):
            return self.select(value)
        else:
            raise TypeError(f"Selection by type {type(value).__name__} is not supported.")

    def get_column(self, name: str) -> ColumnWrapper:
        """
        Retrieves a single column by name, wrapped in a ColumnWrapper.

        Raises:
            KeyError: If the column name does not exist.
        """
        try:
            protocol_col = self._interchange_obj.get_column_by_name(name)
            return ColumnWrapper(protocol_col)
        except KeyError:
            raise KeyError(f"Column '{name}' not found in the dataframe.")

    def select(self, columns: List[str]) -> DataFrameWrapper:
        """
        Selects a subset of columns and returns a new DataFrameWrapper.
        """
        # This operation may involve a copy if the library doesn't support a zero-copy view
        new_interchange_obj = self._interchange_obj.select_columns_by_name(columns)

        # The protocol currently lacks a standard `from_dataframe` method.
        # The most reliable way to reconstruct is to go through a known format.
        # Pandas is the de-facto standard for this temporary conversion.
        temp_pandas_df = pd.api.interchange.from_dataframe(new_interchange_obj)

        # Re-wrap the new dataframe
        return DataFrameWrapper(temp_pandas_df)

    def filter(self, mask: ColumnWrapper) -> DataFrameWrapper:
        """
        Filters rows based on a boolean mask.

        Args:
            mask: A ColumnWrapper containing a boolean series.
        """
        if mask._protocol_col.dtype[0] != 20:  # Dtype ID for BOOL is 20
            raise TypeError("Filter mask must be of boolean type.")

        new_interchange_obj = self._interchange_obj.select_rows_by_mask(mask._protocol_col)
        temp_pandas_df = pd.api.interchange.from_dataframe(new_interchange_obj)
        return DataFrameWrapper(temp_pandas_df)

    def to_numpy(self, columns: List[str] | None = None) -> np.ndarray:
        """
        Converts specified columns (or all) into a 2D NumPy array.
        """
        target_cols = columns or self.columns

        # Convert each column to a NumPy array and stack them horizontally
        numpy_cols = [self.get_column(name).to_numpy() for name in target_cols]

        return np.column_stack(numpy_cols)

    def from_numpy(self, array: np.ndarray, columns: List[str]) -> DataFrameObject:
        """
        Creates a new dataframe of the *original* type from a NumPy array.
        """
        if self._original_type is pd.DataFrame:
            return pd.DataFrame(array, columns=columns)
        if self._original_type is pl.DataFrame:
            return pl.from_numpy(array, schema=columns)

        raise TypeError(f"Construction for dataframe type '{self._original_type.__name__}' is not supported.")


class ColumnWrapper:
    """A detailed wrapper for a single column from an interchange object."""

    def __init__(self, protocol_col: InterchangeColumn):
        self._protocol_col = protocol_col

    @property
    def name(self) -> str:
        """Returns the name of the column."""
        return self._protocol_col.name()

    @property
    def dtype(self) -> Any:
        """Returns the NumPy dtype of the column."""
        return _INTERCHANGE_TO_NUMPY_DTYPE.get(self._protocol_col.dtype[0], np.object_)

    def to_numpy(self) -> np.ndarray:
        """
        Converts the column's data into a single, contiguous NumPy array,
        handling nulls and attempting zero-copy conversion via DLPack.
        """
        numpy_chunks = []
        for chunk in self._protocol_col.get_chunks():
            # Get the main data buffer
            data_buffer = chunk.get_buffers()["data"][0]

            # Use DLPack for zero-copy transfer where supported (e.g., from Polars/cuDF to NumPy)
            if hasattr(data_buffer, "__dlpack__"):
                array_chunk = np.from_dlpack(data_buffer)
            else:  
                # Fallback for protocols not supporting DLPack (like older pandas)
                # This will likely create a copy
                array_chunk = np.frombuffer(data_buffer, dtype=self.dtype)

            # Null Handling
            validity = chunk.get_buffers().get("validity")
            if validity:
                validity_buffer = validity[0]

                null_mask = self._create_mask_from_bitmask(validity_buffer, len(array_chunk))

                # Apply the mask. For floats, we use np.nan. For others, this requires
                # creating an object array or handling it based on context.
                if "float" in array_chunk.dtype.name:
                    array_chunk = array_chunk.astype(np.float64)  # Promote to allow NaN
                    array_chunk[~null_mask] = np.nan
                else:  
                    # For integers, bools, etc., use an object dtype
                    array_chunk = array_chunk.astype(object)
                    array_chunk[~null_mask] = None

            numpy_chunks.append(array_chunk)

        return np.concatenate(numpy_chunks) if numpy_chunks else np.array([], dtype=self.dtype)

    def _create_mask_from_bitmask(self, buffer, size):
        """Helper to convert a validity bitmask into a NumPy boolean array."""
        # Reads the bits from the validity buffer.
        # A 1 indicates a valid value, a 0 indicates a null.
        bytes_array = np.frombuffer(buffer, dtype=np.uint8)
        # Unpack the bits from each byte into a boolean array
        bool_array = np.unpackbits(bytes_array, bitorder="little")
        # Trim to the actual size of the column chunk
        return bool_array[:size]
