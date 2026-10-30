import numpy as np
import pytest

from dowhy.gcm.divergence import auto_estimate_kl_divergence, estimate_kl_divergence_categorical


@pytest.mark.parametrize("estimator", [estimate_kl_divergence_categorical, auto_estimate_kl_divergence])
@pytest.mark.parametrize("dtype", [str, object, bool])
def test_categorical_kl_distinguishes_joint_distributions_with_equal_marginals(estimator, dtype):
    # P(aa, ab, ba, bb) = (3, 1, 1, 3)/8; Q = (1, 3, 3, 1)/8.
    # Both columns have the same marginal probabilities in P and Q, but
    # D(P||Q) = 2*(3/8)*log(3) + 2*(1/8)*log(1/3) = log(3)/2.
    x = np.array([[False, False]] * 3 + [[False, True], [True, False]] + [[True, True]] * 3, dtype=dtype)
    y = np.array([[False, False]] + [[False, True]] * 3 + [[True, False]] * 3 + [[True, True]], dtype=dtype)
    original_x, original_y = x.copy(), y.copy()

    assert estimator(x, y) == pytest.approx(np.log(3) / 2)
    assert estimator(x, np.repeat(y, 2, axis=0)) == pytest.approx(np.log(3) / 2)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("estimator", [estimate_kl_divergence_categorical, auto_estimate_kl_divergence])
def test_categorical_kl_preserves_information_when_duplicating_a_column(estimator):
    x = np.array(["a", "a", "a", "b"])
    y = np.array(["a", "b", "b", "b"])
    expected = np.log(3) / 2

    assert estimator(x, y) == pytest.approx(expected)
    assert estimator(x[:, None], y[:, None]) == pytest.approx(expected)
    assert estimator(np.column_stack([x, x]), np.column_stack([y, y])) == pytest.approx(expected)


@pytest.mark.parametrize("estimator", [estimate_kl_divergence_categorical, auto_estimate_kl_divergence])
def test_categorical_kl_treats_column_positions_as_part_of_the_category(estimator):
    x = np.array([["a", "b"]] * 3 + [["b", "a"]], dtype=object)
    y = np.array([["a", "b"]] + [["b", "a"]] * 3, dtype=object)

    assert estimator(x, y) == pytest.approx(np.log(3) / 2)
    assert estimator(x, x[::-1]) == pytest.approx(0)


def test_categorical_kl_still_rejects_different_feature_counts():
    with pytest.raises(RuntimeError, match="same dimension"):
        estimate_kl_divergence_categorical(np.array([["a", "b"]]), np.array([["a"]]))


@pytest.mark.parametrize("estimator", [estimate_kl_divergence_categorical, auto_estimate_kl_divergence])
def test_categorical_kl_keeps_finite_regularization_for_disjoint_joint_support(estimator):
    x = np.array([["a", "a"], ["b", "b"]])
    y = np.array([["a", "b"], ["b", "a"]])
    eps = np.finfo(np.float64).eps
    # Two categories have counts (1, 0) and two have counts (0, 1).
    expected = 2 / (2 + eps) * np.log((1 + eps) / eps)

    assert np.isfinite(estimator(x, y))
    assert estimator(x, y) == pytest.approx(expected)
