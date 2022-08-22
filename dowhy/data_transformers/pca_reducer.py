from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from dowhy.data_transformer import DimensionalityReducer


class PCAReducer(DimensionalityReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._do_standardize = True
        if "standardize" in kwargs:
            self._do_standardize = kwargs["standardize"]

    def reduce(self):
        data = self._data
        if self._do_standardize:
            data = scale(self._data, axis=0)
        pca_model = PCA(n_components=self._ndims)
        reduced_data = pca_model.fit_transform(data)
        return reduced_data
