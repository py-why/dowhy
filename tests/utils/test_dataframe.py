import pytest
import pandas as pd
import polars as pl
import numpy as np

from dowhy.utils.dataframe import DataFrameWrapper, ColumnWrapper


@pytest.fixture(params=[pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def df_lib(request):
    """Fixture to parameterize tests for both pandas and polars."""
    return request.param


@pytest.fixture
def basic_data():
    """A simple, clean dataset with mixed numeric types."""
    return {
        "A": [1, 2, 3, 4],
        "B": [10.0, 20.0, 30.0, 40.0],
        "C": [100, 200, 300, 400],
    }


@pytest.fixture
def mixed_type_data():
    """A dataset with numeric, boolean, and string data."""
    return {
        "integers": [1, 2, 3],
        "floats": [1.1, 2.2, 3.3],
        "booleans": [True, False, True],
        "strings": ["apple", "banana", "cherry"],
    }


@pytest.fixture
def null_data():
    """A dataset containing various forms of null values."""
    return {
        "int_with_null": [1, None, 3, 4],
        "float_with_null": [10.1, 20.2, np.nan, 40.4],
        "string_with_null": ["a", "b", None, "d"],
    }


@pytest.fixture
def empty_df(df_lib):
    """An empty dataframe."""
    return df_lib()


@pytest.fixture
def single_row_df(df_lib, basic_data):
    """A dataframe with only one row."""
    return df_lib({key: [value[0]] for key, value in basic_data.items()})


@pytest.fixture
def single_col_df(df_lib):
    """A dataframe with only one column."""
    return df_lib({"A": [10, 20, 30]})


# --- Test Suite: Verifying Every Component Under Every Condition ---


class TestDataFrameWrapper:
    """Comprehensive test suite for the DataFrameWrapper."""

    def test_initialization_success(self, df_lib, basic_data):
        """
        VERIFICATION: Ensures the wrapper can be instantiated with a valid
        dataframe object from both pandas and polars.
        """
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        assert wrapper is not None
        assert wrapper._original_type is df_lib

    def test_initialization_failure_unsupported_type(self):
        """
        VERIFICATION: Ensures the wrapper raises a TypeError for objects
        that do not implement the interchange protocol.
        FAILURE MODE: Passing a raw NumPy array or list should fail gracefully.
        """
        with pytest.raises(TypeError, match="does not support the dataframe interchange protocol"):
            DataFrameWrapper(np.array([1, 2, 3]))
        with pytest.raises(TypeError):
            DataFrameWrapper([1, 2, 3])

    # --- Metadata Property Tests ---

    def test_shape_property(self, df_lib, basic_data):
        """VERIFICATION: Checks if the .shape property is correct."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        assert wrapper.shape == (4, 3)

    def test_shape_property_empty(self, empty_df):
        """VERIFICATION: Checks .shape on an empty dataframe."""
        wrapper = DataFrameWrapper(empty_df)
        assert wrapper.shape == (0, 0)

    def test_columns_property(self, df_lib, basic_data):
        """VERIFICATION: Checks if the .columns property returns correct names."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        assert wrapper.columns == ["A", "B", "C"]

    def test_dtypes_property(self, df_lib, mixed_type_data):
        """VERIFICATION: Checks if .dtypes correctly maps protocol types to NumPy types."""
        df = df_lib(mixed_type_data)
        wrapper = DataFrameWrapper(df)
        expected_dtypes = {
            "integers": np.int64,
            "floats": np.float64,
            "booleans": np.bool_,
            "strings": np.object_,
        }
        assert wrapper.dtypes == expected_dtypes

    # --- Column Access and Selection Tests ---

    def test_get_column_success(self, df_lib, basic_data):
        """VERIFICATION: Ensures a single column can be retrieved successfully."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        col_wrapper = wrapper.get_column("B")
        assert isinstance(col_wrapper, ColumnWrapper)
        assert col_wrapper.name == "B"

    def test_getitem_for_get_column(self, df_lib, basic_data):
        """VERIFICATION: Ensures dict-like access `wrapper['col']` works."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        col_wrapper = wrapper["B"]
        assert isinstance(col_wrapper, ColumnWrapper)
        assert col_wrapper.name == "B"

    def test_get_column_failure_nonexistent(self, df_lib, basic_data):
        """
        VERIFICATION: Ensures retrieving a non-existent column raises a KeyError.
        FAILURE MODE: Requesting a column that is not in the dataframe.
        """
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        with pytest.raises(KeyError, match="'X' not found"):
            wrapper.get_column("X")

    def test_select_columns_success(self, df_lib, basic_data):
        """VERIFICATION: Ensures selecting a subset of columns works correctly."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        new_wrapper = wrapper.select(["C", "A"])
        assert isinstance(new_wrapper, DataFrameWrapper)
        assert new_wrapper.columns == ["C", "A"]
        assert new_wrapper.shape == (4, 2)

    # --- Critical ColumnWrapper Tests ---

    def test_column_to_numpy_numeric(self, df_lib, basic_data):
        """VERIFICATION: Converts a clean numeric column to a NumPy array."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        col_numpy = wrapper.get_column("A").to_numpy()
        assert isinstance(col_numpy, np.ndarray)
        np.testing.assert_array_equal(col_numpy, np.array([1, 2, 3, 4]))
        assert col_numpy.dtype == np.int64

    def test_column_to_numpy_with_nulls(self, df_lib, null_data):
        """
        VERIFICATION: Correctly handles nulls when converting to NumPy.
        - Floats should use np.nan.
        - Integers and others should use None, resulting in an 'object' dtype.
        """
        if isinstance(null_data, dict) and df_lib == pd.DataFrame:
            # Pandas needs explicit dtype for integer arrays with NaNs
            null_data["int_with_null"] = pd.Series(null_data["int_with_null"], dtype="Int64")

        df = df_lib(null_data)
        wrapper = DataFrameWrapper(df)

        # Test float column with np.nan
        float_col = wrapper.get_column("float_with_null").to_numpy()
        np.testing.assert_allclose(float_col, np.array([10.1, 20.2, np.nan, 40.4]), equal_nan=True)
        assert "float" in float_col.dtype.name

        # Test integer column with None
        int_col = wrapper.get_column("int_with_null").to_numpy()
        assert int_col.dtype == object
        assert int_col[1] is None
        np.testing.assert_array_equal(int_col[[0, 2, 3]], np.array([1, 3, 4]))

    # --- Full DataFrame Conversion and Reconstruction ---

    def test_to_numpy_full_df(self, df_lib, basic_data):
        """VERIFICATION: Converts an entire dataframe to a 2D NumPy array."""
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        full_array = wrapper.to_numpy()
        expected_array = df.to_numpy()
        np.testing.assert_array_equal(full_array, expected_array)
        assert full_array.shape == (4, 3)

    def test_from_numpy_reconstruction(self, df_lib, basic_data):
        """
        VERIFICATION: Reconstructs a dataframe of the original type from a NumPy array.
        This is critical for ensuring outputs match user inputs.
        """
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        numpy_array = wrapper.to_numpy()

        reconstructed_df = wrapper.from_numpy(numpy_array, columns=["A", "B", "C"])

        assert isinstance(reconstructed_df, df_lib)
        # For Polars, comparison is best done after converting both to a common format
        if df_lib == pl.DataFrame:
            assert reconstructed_df.to_pandas().equals(df.to_pandas())
        else:
            assert reconstructed_df.equals(df)

    def test_filter_success(self, df_lib, mixed_type_data):
        """VERIFICATION: Filters rows correctly using a boolean mask."""
        df = df_lib(mixed_type_data)
        wrapper = DataFrameWrapper(df)

        mask_col = wrapper.get_column("booleans")
        filtered_wrapper = wrapper.filter(mask_col)

        assert filtered_wrapper.shape == (2, 4)
        # Verify content of a column in the filtered dataframe
        filtered_strings = filtered_wrapper.get_column("strings").to_numpy()
        np.testing.assert_array_equal(filtered_strings, np.array(["apple", "cherry"]))

    def test_filter_failure_non_boolean_mask(self, df_lib, basic_data):
        """
        VERIFICATION: Raises TypeError when attempting to filter with a non-boolean column.
        FAILURE MODE: Using a numeric or string column as a filter mask.
        """
        df = df_lib(basic_data)
        wrapper = DataFrameWrapper(df)
        numeric_mask = wrapper.get_column("A")

        with pytest.raises(TypeError, match="Filter mask must be of boolean type"):
            wrapper.filter(numeric_mask)
