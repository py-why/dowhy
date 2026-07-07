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


# ── _split_by_attribute (new helper from PR #1371) ────────────────────────────


def test_split_by_attribute_two_labels():
    """_split_by_attribute correctly splits features into two groups."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    labels = torch.tensor([0, 1, 0, 1])
    result = reg._split_by_attribute(features, labels)

    assert len(result) == 2
    assert result[0].tolist() == [[1.0, 2.0], [5.0, 6.0]]  # label == 0
    assert result[1].tolist() == [[3.0, 4.0], [7.0, 8.0]]  # label == 1


def test_split_by_attribute_single_label_returns_full_tensor():
    """_split_by_attribute returns a one-element list when all labels are identical."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([0, 0])
    result = reg._split_by_attribute(features, labels)

    assert len(result) == 1
    assert result[0].tolist() == features.tolist()


def test_split_by_attribute_three_labels():
    """_split_by_attribute handles three distinct label values."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    features = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    result = reg._split_by_attribute(features, labels)

    assert len(result) == 3
    # Each group should have 2 samples
    for group in result:
        assert group.shape[0] == 2


# ── _optimized_mmd_penalty (new method from PR #1371) ────────────────────────


def test_optimized_mmd_penalty_single_tensor_returns_zero():
    """_optimized_mmd_penalty returns 0.0 for a single-tensor list (nothing to compare)."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    result = reg._optimized_mmd_penalty([torch.randn(8, 4)])
    assert result == 0.0


def test_optimized_mmd_penalty_empty_tensors_skipped():
    """_optimized_mmd_penalty ignores zero-row tensors; returns 0.0 when nothing remains."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    result = reg._optimized_mmd_penalty([torch.zeros(0, 4), torch.zeros(0, 4)])
    assert result == 0.0


def test_optimized_mmd_penalty_gaussian_matches_naive_two_tensors():
    """_optimized_mmd_penalty (gaussian) equals naive mmd() for two tensors."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(7)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=0.5)
    t1, t2 = torch.randn(16, 4), torch.randn(16, 4)

    optimized = float(reg._optimized_mmd_penalty([t1, t2]))
    naive = float(reg.mmd(t1, t2))

    assert optimized == pytest.approx(naive, rel=1e-4)


def test_optimized_mmd_penalty_gaussian_matches_naive_three_tensors():
    """_optimized_mmd_penalty (gaussian) sums all pairwise MMDs correctly for three tensors."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(11)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    t1, t2, t3 = torch.randn(12, 3), torch.randn(12, 3), torch.randn(12, 3)

    optimized = float(reg._optimized_mmd_penalty([t1, t2, t3]))
    naive = float(reg.mmd(t1, t2) + reg.mmd(t1, t3) + reg.mmd(t2, t3))

    assert optimized == pytest.approx(naive, rel=1e-4)


def test_optimized_mmd_penalty_linear_kernel_matches_naive():
    """_optimized_mmd_penalty (non-gaussian path) matches the naive mmd() call."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(3)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="linear", gamma=1.0)
    t1, t2 = torch.randn(10, 4), torch.randn(10, 4)

    optimized = float(reg._optimized_mmd_penalty([t1, t2]))
    naive = float(reg.mmd(t1, t2))

    assert optimized == pytest.approx(naive, rel=1e-4)


def test_optimized_mmd_penalty_fp64_precision_small_gamma():
    """
    PR #1371 numeric-stability fix: float64 accumulation inside _optimized_mmd_penalty
    prevents the penalty from vanishing when gamma is very small.
    """
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1e-6)
    # Use clearly separated tensors to guarantee a non-zero penalty.
    t1 = torch.zeros(8, 4)
    t2 = torch.ones(8, 4) * 100.0

    penalty = float(reg._optimized_mmd_penalty([t1, t2]))

    assert torch.isfinite(torch.tensor(penalty))
    assert penalty > 0.0


# ── unconditional_reg branches ────────────────────────────────────────────────


def test_unconditional_reg_e_eq_a_true_e_conditioned_false():
    """unconditional_reg(E_eq_A=True, E_conditioned=False) sums pairwise MMD between envs."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    c0 = torch.zeros(8, 4)
    c1 = torch.ones(8, 4) * 5.0
    c2 = torch.ones(8, 4) * 10.0
    attr_labels = [torch.zeros(8, dtype=torch.long)] * 3  # attr value unused in this branch

    penalty = float(reg.unconditional_reg([c0, c1, c2], attr_labels, num_envs=3, E_eq_A=True))

    # All three environments are clearly different → positive pairwise MMD.
    assert penalty > 0.0


def test_unconditional_reg_e_eq_a_true_e_conditioned_true_returns_zero():
    """unconditional_reg(E_eq_A=True, E_conditioned=True) applies no penalty (branch not taken)."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    classifs = [torch.randn(8, 4), torch.randn(8, 4)]
    attr_labels = [torch.zeros(8, dtype=torch.long)] * 2

    penalty = float(reg.unconditional_reg(classifs, attr_labels, num_envs=2, E_eq_A=True))

    assert penalty == pytest.approx(0.0)


def test_unconditional_reg_e_conditioned_true_splits_within_env():
    """unconditional_reg(E_conditioned=True, E_eq_A=False) penalizes within-env attr differences."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    # Each env has two attr values with clearly different feature distributions.
    n = 8
    c0 = torch.cat([torch.zeros(n, 4), torch.ones(n, 4) * 5.0])
    attr0 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    c1 = c0.clone() + 0.1
    attr1 = attr0.clone()

    penalty = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False))

    assert penalty > 0.0


def test_unconditional_reg_not_e_conditioned_same_attr_gives_no_penalty():
    """
    Regression test for PR #1371: when all samples share the same attribute value across
    environments, unconditional_reg(E_conditioned=False) must return zero.

    The old code erroneously produced a positive penalty by treating per-environment attribute
    groups as distinct even when the attribute value was identical.
    """
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(0)
    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    # Both environments have only attr=0 → after concatenation there is a single attr group.
    c0 = torch.randn(10, 4)
    c1 = torch.randn(10, 4)
    attr0 = torch.zeros(10, dtype=torch.long)
    attr1 = torch.zeros(10, dtype=torch.long)

    penalty = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False))

    assert penalty == pytest.approx(0.0, abs=1e-6)


def test_unconditional_reg_not_e_conditioned_different_attrs_positive_penalty():
    """unconditional_reg(E_conditioned=False) returns a positive penalty when attr values differ."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    # env0 has attr=0 with zero features; env1 has attr=1 with large-valued features.
    n = 10
    c0 = torch.zeros(n, 4)
    c1 = torch.ones(n, 4) * 10.0
    attr0 = torch.zeros(n, dtype=torch.long)  # attr=0
    attr1 = torch.ones(n, dtype=torch.long)  # attr=1

    penalty = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False))

    assert penalty > 0.0


def test_unconditional_reg_optimization_matches_naive_e_conditioned_true():
    """unconditional_reg: optimized and naive paths agree for E_conditioned=True."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(42)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    c0 = torch.cat([torch.randn(n, 4), torch.randn(n, 4) + 3.0])
    attr0 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    c1 = torch.cat([torch.randn(n, 4) + 1.0, torch.randn(n, 4) + 4.0])
    attr1 = attr0.clone()

    naive = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False, use_optimization=False))
    optimized = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False, use_optimization=True))

    assert optimized == pytest.approx(naive, rel=1e-4)


def test_unconditional_reg_optimization_matches_naive_not_e_conditioned():
    """unconditional_reg: optimized and naive paths agree for E_conditioned=False."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(5)
    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 10
    c0, c1 = torch.randn(n, 3), torch.randn(n, 3)
    attr0 = torch.randint(0, 2, (n,))
    attr1 = torch.randint(0, 2, (n,))

    naive = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False, use_optimization=False))
    optimized = float(reg.unconditional_reg([c0, c1], [attr0, attr1], num_envs=2, E_eq_A=False, use_optimization=True))

    assert optimized == pytest.approx(naive, rel=1e-4)


# ── _compute_conditional_penalty ─────────────────────────────────────────────


def test_compute_conditional_penalty_single_group_multiple_attrs():
    """_compute_conditional_penalty penalizes different attribute values within one group."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    # All samples belong to group 0; two attr values with clearly different features.
    features = torch.cat([torch.zeros(n, 4), torch.ones(n, 4) * 5.0])
    attributes = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    group_data = torch.zeros(2 * n, 1)  # single group

    penalty = float(reg._compute_conditional_penalty(features, attributes, group_data, use_optimization=False))

    assert penalty > 0.0


def test_compute_conditional_penalty_each_group_has_one_attr():
    """_compute_conditional_penalty returns zero when every group contains only one attr value."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    # Group 0: all attr=0; Group 1: all attr=1 → no cross-attr MMD within any group.
    features = torch.randn(2 * n, 4)
    attributes = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    group_data = torch.cat([torch.zeros(n, 1), torch.ones(n, 1)])

    penalty = float(reg._compute_conditional_penalty(features, attributes, group_data, use_optimization=False))

    assert penalty == pytest.approx(0.0, abs=1e-6)


def test_compute_conditional_penalty_value_based_grouping():
    """
    Key regression test for PR #1371: group_data rows with equal VALUES must be merged into the
    same group regardless of which Python object they originated from.

    The old implementation used PyTorch tensors as dictionary keys (identity-based hashing).
    Equal-valued tensors created in different environments were treated as distinct keys,
    producing an erroneously large penalty.  The new code uses torch.unique(dim=0) for
    value-based row comparison.
    """
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    n = 8
    # env0 samples: attr=0, near-zero features.
    # env1 samples: attr=1, large-valued features.
    # Both sets have group_data == 0 (same group), created as independent tensors.
    feats_env0 = torch.zeros(n, 4)
    feats_env1 = torch.ones(n, 4) * 5.0
    features = torch.cat([feats_env0, feats_env1])
    attrs = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    # Two independently-created zero-tensors: same value, different Python objects.
    gd_env0 = torch.zeros(n, 1, dtype=torch.float32)
    gd_env1 = torch.zeros(n, 1, dtype=torch.float32)
    assert gd_env0.data_ptr() != gd_env1.data_ptr(), "Precondition: must be different objects"
    group_data = torch.cat([gd_env0, gd_env1])

    # With value-based grouping all 2n samples fall into group 0.
    # Group 0 contains attr=0 (zeros) and attr=1 (fives) → MMD > 0.
    penalty = float(reg._compute_conditional_penalty(features, attrs, group_data, use_optimization=False))

    assert penalty > 0.0


def test_compute_conditional_penalty_optimization_matches_naive():
    """_compute_conditional_penalty: optimized path gives the same result as the naive path."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(99)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    features = torch.cat([torch.randn(n, 4), torch.randn(n, 4) + 4.0] * 2)
    attributes = torch.tensor([0] * n + [1] * n + [0] * n + [1] * n, dtype=torch.long)
    group_data = torch.cat([torch.zeros(2 * n, 1), torch.ones(2 * n, 1)])

    naive = float(reg._compute_conditional_penalty(features, attributes, group_data, use_optimization=False))
    optimized = float(reg._compute_conditional_penalty(features, attributes, group_data, use_optimization=True))

    assert optimized == pytest.approx(naive, rel=1e-4)


# ── conditional_reg branches ──────────────────────────────────────────────────


def test_conditional_reg_e_conditioned_true_penalizes_within_env():
    """conditional_reg(E_conditioned=True) applies per-environment conditional grouping."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    # Each env: within Y=0, attr=0→zeros and attr=1→fives (big MMD).
    c0 = torch.cat([torch.zeros(n, 4), torch.ones(n, 4) * 5.0] * 2)
    attr0 = torch.tensor([0] * n + [1] * n + [0] * n + [1] * n, dtype=torch.long)
    y0 = torch.tensor([0] * n + [0] * n + [1] * n + [1] * n, dtype=torch.long)

    penalty = float(reg.conditional_reg([c0], [attr0], [[y0]], num_envs=1, E_eq_A=False))

    assert penalty > 0.0


def test_conditional_reg_not_e_conditioned_cross_env_merging():
    """
    Regression test for PR #1371: conditional_reg(E_conditioned=False) correctly merges samples
    from different environments that share the same conditioning variable values.

    Previously, tensor-as-dict-key bugs caused cross-environment merging to fail; the same
    conditioning value from env0 and env1 was stored under separate dict entries, inflating
    the penalty incorrectly.
    """
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    # 2 environments, each with Y ∈ {0, 1}.
    # env0: all samples have attr=0 and near-zero features.
    # env1: all samples have attr=1 and large features.
    # After cross-env merging, group Y=0 contains both attr=0 (env0) and attr=1 (env1) → MMD > 0.
    n = 8
    c0 = torch.zeros(2 * n, 4)
    c1 = torch.ones(2 * n, 4) * 5.0
    attr0 = torch.zeros(2 * n, dtype=torch.long)
    attr1 = torch.ones(2 * n, dtype=torch.long)
    y0 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    y1 = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    penalty = float(reg.conditional_reg([c0, c1], [attr0, attr1], [[y0, y1]], num_envs=2, E_eq_A=False))

    assert penalty > 0.0


def test_conditional_reg_returns_zero_when_group_has_single_attr():
    """conditional_reg returns zero when every conditioning group has only one attribute value."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    n = 8
    c0 = torch.randn(n, 4)
    c1 = torch.randn(n, 4)
    attr0 = torch.zeros(n, dtype=torch.long)  # all attr=0 in env0
    attr1 = torch.zeros(n, dtype=torch.long)  # all attr=0 in env1
    y0 = torch.zeros(n, dtype=torch.long)
    y1 = torch.zeros(n, dtype=torch.long)

    # After merging, group Y=0 has only attr=0 → no cross-attr MMD.
    penalty = float(reg.conditional_reg([c0, c1], [attr0, attr1], [[y0, y1]], num_envs=2, E_eq_A=False))

    assert penalty == pytest.approx(0.0, abs=1e-6)


def test_conditional_reg_optimization_matches_naive():
    """conditional_reg: optimized path gives the same result as the naive path."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    torch.manual_seed(123)
    reg = Regularizer(E_conditioned=True, ci_test="mmd", kernel_type="gaussian", gamma=1.0)
    n = 8
    c0 = torch.cat([torch.randn(n, 4), torch.randn(n, 4) + 3.0] * 2)
    attr0 = torch.tensor([0] * n + [1] * n + [0] * n + [1] * n, dtype=torch.long)
    y0 = torch.tensor([0] * n + [0] * n + [1] * n + [1] * n, dtype=torch.long)

    naive = float(reg.conditional_reg([c0], [attr0], [[y0]], num_envs=1, E_eq_A=False, use_optimization=False))
    optimized = float(reg.conditional_reg([c0], [attr0], [[y0]], num_envs=1, E_eq_A=False, use_optimization=True))

    assert optimized == pytest.approx(naive, rel=1e-4)


def test_conditional_reg_e_eq_a_true_not_e_conditioned():
    """conditional_reg(E_eq_A=True, E_conditioned=False) pools features by env and conditions."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.algorithms.regularization import Regularizer

    reg = Regularizer(E_conditioned=False, ci_test="mmd", kernel_type="gaussian", gamma=1.0)

    # 2 environments with different feature magnitudes; same Y conditioning values.
    n = 8
    c0 = torch.zeros(n, 4)
    c1 = torch.ones(n, 4) * 5.0
    attr0 = torch.zeros(n, dtype=torch.long)
    attr1 = torch.zeros(n, dtype=torch.long)
    y0 = torch.zeros(n, dtype=torch.long)
    y1 = torch.zeros(n, dtype=torch.long)

    # E_eq_A=True: attribute coincides with environment → each env's data treated as one attr group.
    penalty = float(reg.conditional_reg([c0, c1], [attr0, attr1], [[y0, y1]], num_envs=2, E_eq_A=True))

    # env0 (attr==0) and env1 (attr==1) are in the same group and have different features → > 0.
    assert penalty > 0.0


# ── CACM fixes from PR #1371 ──────────────────────────────────────────────────


def test_cacm_attr_types_none_default():
    """CACM initialises without error using the new None default for attr_types."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = CACM(model)

    assert algo.attr_types is None
    assert algo.E_eq_A is None


def test_cacm_no_mutable_default_aliasing():
    """Two CACM instances without explicit attr_types/E_eq_A are independently None (no aliasing)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    algo_a = CACM(torch.nn.Sequential(torch.nn.Linear(4, 2)))
    algo_b = CACM(torch.nn.Sequential(torch.nn.Linear(4, 2)))

    # With mutable default arguments (old code: attr_types=[]) both instances share the same
    # list object; with None defaults (new code) they are independently None.
    assert algo_a.attr_types is None
    assert algo_b.attr_types is None


def test_cacm_e_eq_a_none_does_not_raise(monkeypatch):
    """
    Regression test for PR #1371: CACM.training_step must not raise when E_eq_A=None.

    The old code evaluated `attr_type_idx in self.E_eq_A` unconditionally; with E_eq_A=None
    this raises TypeError.  The fix guards with `False if self.E_eq_A is None else ...`.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = CACM(model, attr_types=["conf"], E_eq_A=None)
    monkeypatch.setattr(algo, "log_dict", lambda *a, **kw: None)
    monkeypatch.setattr(algo, "log", lambda *a, **kw: None)

    n = 16
    x, y = torch.randn(n, 4), torch.randint(0, 2, (n,))
    a = torch.randint(0, 2, (n, 1)).float()
    batch = [(x, y, a), (x.clone(), y.clone(), a.clone())]

    loss = algo.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_cacm_training_step_causal_attr_type(monkeypatch):
    """CACM.training_step with attr_types=['causal'] produces a finite loss on synthetic data."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = CACM(model, attr_types=["causal"], E_eq_A=None, gamma=1.0)
    monkeypatch.setattr(algo, "log_dict", lambda *a, **kw: None)
    monkeypatch.setattr(algo, "log", lambda *a, **kw: None)

    n = 16
    x, y = torch.randn(n, 4), torch.randint(0, 2, (n,))
    a = torch.randint(0, 2, (n, 1)).float()
    batch = [(x, y, a), (x.clone(), y.clone(), a.clone())]

    loss = algo.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_cacm_training_step_all_attr_types(monkeypatch):
    """CACM.training_step with all four attr_types produces a finite total loss."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = CACM(model, attr_types=["causal", "conf", "ind", "sel"], E_eq_A=None, gamma=1.0)
    monkeypatch.setattr(algo, "log_dict", lambda *a, **kw: None)
    monkeypatch.setattr(algo, "log", lambda *a, **kw: None)

    n = 16
    x, y = torch.randn(n, 4), torch.randint(0, 2, (n,))
    # 4 attribute columns, one per attr_type.
    a = torch.randint(0, 2, (n, 4)).float()
    batch = [(x, y, a), (x.clone(), y.clone(), a.clone())]

    loss = algo.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_cacm_training_step_with_e_eq_a_flag(monkeypatch):
    """CACM.training_step runs correctly when E_eq_A=[0] marks attribute 0 as env-coincident."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.cacm import CACM

    torch.manual_seed(2)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = CACM(model, attr_types=["conf"], E_eq_A=[0], gamma=1.0)
    monkeypatch.setattr(algo, "log_dict", lambda *a, **kw: None)
    monkeypatch.setattr(algo, "log", lambda *a, **kw: None)

    n = 16
    x, y = torch.randn(n, 4), torch.randint(0, 2, (n,))
    a = torch.randint(0, 2, (n, 1)).float()
    batch = [(x, y, a), (x.clone(), y.clone(), a.clone())]

    loss = algo.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


# ── base_algorithm: AdamW added in PR #1371 ───────────────────────────────────


def test_prediction_algorithm_accepts_adamw():
    """PredictionAlgorithm accepts 'AdamW' without raising (PR #1371 addition)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.base_algorithm import PredictionAlgorithm

    model = torch.nn.Linear(4, 2)
    algo = PredictionAlgorithm(model, optimizer="AdamW", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9)

    assert algo.optimizer == "AdamW"


def test_erm_configure_optimizers_adamw():
    """ERM.configure_optimizers returns an AdamW instance when optimizer='AdamW'."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.erm import ERM

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    algo = ERM(model, optimizer="AdamW", lr=0.01)
    opt = algo.configure_optimizers()

    assert isinstance(opt, torch.optim.AdamW)


# ── ERM end-to-end with synthetic data ───────────────────────────────────────


def test_erm_training_step_with_synthetic_data(monkeypatch):
    """ERM.training_step produces a finite, positive loss on small synthetic data."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from dowhy.causal_prediction.algorithms.erm import ERM

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    algo = ERM(model, optimizer="Adam", lr=1e-3)
    monkeypatch.setattr(algo, "log_dict", lambda *a, **kw: None)

    n = 16
    x, y = torch.randn(n, 4), torch.randint(0, 2, (n,))
    a = torch.zeros(n, 1)

    # ERM receives a list of (x, y, a) tuples — one per environment.
    batch = [(x, y, a), (x.clone(), y.clone(), a.clone())]
    loss = algo.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert float(loss) > 0.0


# ── dataloader utilities ──────────────────────────────────────────────────────


def test_split_dataset_sizes():
    """split_dataset produces two complementary subsets of the requested sizes."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import split_dataset

    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.data = list(range(n))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ds = _TinyDataset(20)
    val, train = split_dataset(ds, n=5, seed=42)

    assert len(val) == 5
    assert len(train) == 15
    assert len(val) + len(train) == len(ds)


def test_split_dataset_is_reproducible():
    """split_dataset produces the same split for the same seed."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import split_dataset

    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.data = list(range(n))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ds = _TinyDataset(20)
    val_a, _ = split_dataset(ds, n=5, seed=0)
    val_b, _ = split_dataset(ds, n=5, seed=0)

    assert [val_a[i] for i in range(len(val_a))] == [val_b[i] for i in range(len(val_b))]


def test_split_dataset_different_seeds_differ():
    """split_dataset produces different splits for different seeds (with high probability)."""
    torch = pytest.importorskip("torch")
    from dowhy.causal_prediction.dataloaders.misc import split_dataset

    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.data = list(range(n))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ds = _TinyDataset(100)
    val_0, _ = split_dataset(ds, n=10, seed=0)
    val_1, _ = split_dataset(ds, n=10, seed=1)

    items_0 = [val_0[i] for i in range(len(val_0))]
    items_1 = [val_1[i] for i in range(len(val_1))]
    assert items_0 != items_1
