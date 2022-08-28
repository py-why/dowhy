import numpy as np
import pytest
from matplotlib import pyplot as plt

from dowhy.data_transformers.pca_reducer import PCAReducer


@pytest.fixture()
def feature_matrix():
    num_features = 2
    num_samples = 10000
    means = np.random.uniform(-1, 1, num_features)
    # cov_mat = np.diag(np.ones(num_features))
    cov_mat = np.ones((num_features, num_features))
    cov_mat[0, 1] = 0.5
    cov_mat[1, 0] = 0.5
    X = np.random.multivariate_normal(means, cov_mat, num_samples)
    return X


class TestPCAReducer:
    def test_reduce(self, feature_matrix, show_plot=False):
        reducer = PCAReducer(feature_matrix, ndims=2, standardize=False)
        X_pca = reducer.reduce()
        print(feature_matrix)
        print(X_pca)
        if show_plot:
            plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c="g")
            plt.xlabel("Dim 0")
            plt.ylabel("Dim 1")
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            plt.show()

        dist_origin = np.square(feature_matrix[:, 0] - np.mean(feature_matrix[:, 0])) + np.square(
            feature_matrix[:, 1] - np.mean(feature_matrix[:, 1])
        )
        dist_origin_pca = np.square(X_pca[:, 0] - np.mean(X_pca[:, 0])) + np.square(X_pca[:, 1] - np.mean(X_pca[:, 1]))
        print((dist_origin_pca - dist_origin) / dist_origin)
        assert all(abs(dist_origin_pca - dist_origin)) < 0.001
