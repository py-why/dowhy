import torch

from dowhy.causal_prediction.algorithms.utils import gaussian_kernel, mmd_compute, my_cdist


class TestAlgorithmUtils:
    def test_my_cdist(self):
        # Squared Euclidean distances between x1 and x2
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x2 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        distances = my_cdist(x1, x2)
        expected = torch.tensor([[1.0, 1.0], [13.0, 5.0]])
        assert torch.allclose(distances, expected, rtol=1e-5)

        # Single vector case
        x1 = torch.tensor([[1.0, 2.0]])
        x2 = torch.tensor([[1.0, 1.0]])
        distances = my_cdist(x1, x2)
        expected = torch.tensor([[1.0]])
        assert torch.allclose(distances, expected, rtol=1e-5)

    def test_gaussian_kernel(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        gamma = 1.0
        kernel = gaussian_kernel(x, y, gamma)

        # Kernel values are exp(-gamma * squared distance)
        assert kernel.shape == (2, 2)
        assert torch.all(kernel >= 0) and torch.all(kernel <= 1)

        # Symmetry for same input
        kernel_xx = gaussian_kernel(x, x, gamma)
        assert torch.allclose(kernel_xx, kernel_xx.t(), rtol=1e-5)

    def test_mmd_compute(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        gamma = 1.0

        # MMD^2 = mean(K(x, x)) + mean(K(y, y)) - 2 * mean(K(x, y))
        mmd_gaussian = mmd_compute(x, y, "gaussian", gamma)
        assert mmd_gaussian >= 0

        # MMD for identical distributions should be zero
        mmd_same = mmd_compute(x, x, "gaussian", gamma)
        assert torch.allclose(mmd_same, torch.tensor(0.0), rtol=1e-5)

        # 'other' kernel: sum of mean squared difference of means and covariances
        mmd_other = mmd_compute(x, y, "other", gamma)
        assert mmd_other >= 0
