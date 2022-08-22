import pytest

from dowhy.data_transformer import DimensionalityReducer


class MockReducer(DimensionalityReducer):
    pass


def test_dimensionality_reducer_placeholder_methods():
    reducer = MockReducer(None, None)
    with pytest.raises(NotImplementedError):
        reducer.reduce()
