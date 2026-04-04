"""Tests for the dowhy.causal_prediction module.

Tests that require torch / pytorch-lightning are automatically skipped when
those packages are not installed.
"""

import pytest

# ── pure-Python tests (no torch / pytorch-lightning required) ─────────────────


def test_multiple_domain_dataset_class_defaults():
    """MultipleDomainDataset exposes the expected class-level defaults."""
    from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset

    assert MultipleDomainDataset.N_STEPS == 5001
    assert MultipleDomainDataset.CHECKPOINT_FREQ == 100
    assert MultipleDomainDataset.N_WORKERS == 8
    assert MultipleDomainDataset.ENVIRONMENTS is None
    assert MultipleDomainDataset.INPUT_SHAPE is None


def test_multiple_domain_dataset_subclass_len_and_getitem():
    """A minimal MultipleDomainDataset subclass returns correct length and items."""
    from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset

    class ToyDataset(MultipleDomainDataset):
        ENVIRONMENTS = ["env_0", "env_1"]

        def __init__(self):
            self.datasets = [[10, 20], [30, 40]]

    ds = ToyDataset()

    assert len(ds) == 2
    assert ds[0] == [10, 20]
    assert ds[1] == [30, 40]


def test_multiple_domain_dataset_subclass_overrides_defaults():
    """Subclass overrides of N_STEPS and ENVIRONMENTS are preserved."""
    from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset

    class CustomDataset(MultipleDomainDataset):
        N_STEPS = 1000
        CHECKPOINT_FREQ = 50
        ENVIRONMENTS = ["a", "b", "c"]
        INPUT_SHAPE = (1, 28, 28)

        def __init__(self):
            self.datasets = [None, None, None]

    ds = CustomDataset()

    assert ds.N_STEPS == 1000
    assert ds.CHECKPOINT_FREQ == 50
    assert ds.ENVIRONMENTS == ["a", "b", "c"]
    assert ds.INPUT_SHAPE == (1, 28, 28)
    assert len(ds) == 3


# ── torch-dependent tests (skipped when torch / pl are absent) ────────────────


def test_seed_hash_deterministic():
    """seed_hash returns the same integer for the same arguments."""
    pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import seed_hash

    assert seed_hash(42, "hello") == seed_hash(42, "hello")


def test_seed_hash_different_inputs_differ():
    """seed_hash returns different values for different arguments."""
    pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import seed_hash

    assert seed_hash(0, "a") != seed_hash(0, "b")
    assert seed_hash(0, "a") != seed_hash(1, "a")


def test_seed_hash_output_in_valid_range():
    """seed_hash output is a non-negative integer strictly less than 2**31."""
    pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import seed_hash

    v = seed_hash(0, 1, "test")
    assert isinstance(v, int)
    assert 0 <= v < 2**31


def test_prediction_algorithm_rejects_unsupported_optimizer():
    """PredictionAlgorithm raises an exception for unsupported optimizer names."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.base_algorithm import PredictionAlgorithm

    model = torch.nn.Linear(4, 2)
    with pytest.raises(Exception, match="not implemented"):
        PredictionAlgorithm(model, optimizer="Adagrad", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9)


def test_erm_init_stores_config():
    """ERM initialises correctly and stores all configuration attributes."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.erm import ERM

    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
    algo = ERM(model, optimizer="Adam", lr=5e-4, weight_decay=1e-4)

    assert algo.model is model
    assert algo.optimizer == "Adam"
    assert algo.lr == pytest.approx(5e-4)
    assert algo.weight_decay == pytest.approx(1e-4)


def test_erm_configure_optimizers_adam():
    """ERM.configure_optimizers returns an Adam optimiser when requested."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.erm import ERM

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    algo = ERM(model, optimizer="Adam", lr=0.01)
    opt = algo.configure_optimizers()

    assert isinstance(opt, torch.optim.Adam)


def test_erm_configure_optimizers_sgd():
    """ERM.configure_optimizers returns an SGD optimiser when requested."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.erm import ERM

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    algo = ERM(model, optimizer="SGD", lr=0.1, momentum=0.9)
    opt = algo.configure_optimizers()

    assert isinstance(opt, torch.optim.SGD)


def test_mmd_compute_zero_for_identical_inputs():
    """mmd_compute with identical tensors returns zero (or near-zero) MMD."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.utils import mmd_compute

    torch.manual_seed(0)
    x = torch.randn(16, 4)
    mmd = mmd_compute(x, x, kernel_type="gaussian", gamma=1.0)

    assert float(mmd) == pytest.approx(0.0, abs=1e-5)


def test_mmd_compute_positive_for_shifted_distributions():
    """mmd_compute returns a positive value when distributions are clearly different."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.utils import mmd_compute

    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.randn(32, 4) + 10.0  # large shift guarantees non-zero MMD
    mmd_gaussian = mmd_compute(x, y, kernel_type="gaussian", gamma=1.0)
    mmd_linear = mmd_compute(x, y, kernel_type="linear", gamma=1.0)

    assert float(mmd_gaussian) > 0.0
    assert float(mmd_linear) > 0.0


def test_regularizer_stores_init_params():
    """Regularizer stores all initialisation parameters as attributes."""
    pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1e-6)

    assert reg.E_conditioned is True
    assert reg.ci_test == "mmd"
    assert reg.kernel_type == "gaussian"
    assert reg.gamma == pytest.approx(1e-6)


def test_regularizer_mmd_zero_for_identical_inputs():
    """Regularizer.mmd returns near-zero for identical input tensors."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(0)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    x = torch.randn(16, 4)
    assert float(reg.mmd(x, x)) == pytest.approx(0.0, abs=1e-5)
