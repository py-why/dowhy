class DimensionalityReducer:
    def __init__(self, data_array, ndims, **kwargs):
        self._data = data_array
        self._ndims = ndims

    def reduce(self, target_dimensions=None):
        raise NotImplementedError
