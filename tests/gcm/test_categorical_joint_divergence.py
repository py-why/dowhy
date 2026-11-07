from decimal import Decimal
from itertools import permutations

import numpy as np
import pytest

from dowhy.gcm.divergence import auto_estimate_kl_divergence, estimate_kl_divergence_categorical


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("kind", ["dates", "durations"])
def test_categorical_kl_handles_mixed_temporal_units_outside_common_range(joint, kind):
    if kind == "dates":
        values = [
            np.datetime64("1500-01-01", "D"),
            np.datetime64("2500-01-01", "D"),
            np.datetime64("2000-01-01", "ns"),
        ]
    else:
        values = [np.timedelta64(-200000, "D"), np.timedelta64(200000, "D"), np.timedelta64(0, "ns")]
    for order in permutations(values):
        x = np.array([["tag", value] for value in order] if joint else order, dtype=object)
        assert estimate_kl_divergence_categorical(x, x.copy()) == pytest.approx(0)
        assert estimate_kl_divergence_categorical(x, x[::-1]) == pytest.approx(0)


@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
@pytest.mark.parametrize("count", [-1, 1])
def test_categorical_kl_keeps_large_stepped_temporal_values_distinct(kind, count):
    value = np.array(count, dtype=f"{kind}[1000000D]")[()]
    x = np.array([value], dtype=object)
    equivalent = np.array([np.array(count * 1000000, dtype=f"{kind}[D]")[()]], dtype=object)
    # This timestamp fits in nanoseconds but differs from the million-day value, despite overflowing comparisons
    # treating them as equal on older NumPy versions.
    wrapped = np.array([np.array(count * -5833720368547758080, dtype=f"{kind}[ns]")[()]], dtype=object)
    eps = np.finfo(float).eps
    expected = np.log((1 + eps) / eps) / (1 + eps)

    assert estimate_kl_divergence_categorical(x, equivalent) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(x, wrapped) == pytest.approx(expected)


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


def test_categorical_kl_treats_floating_nans_as_one_missing_category():
    eps = np.finfo(np.float64).eps
    x = np.array([1.0, 1.0, np.nan, np.nan])
    y = np.array([1.0, np.nan, np.nan, np.nan])
    # P(1, NaN) = (2, 2)/4 and Q(1, NaN) = (1, 3)/4.
    p = (np.array([2, 2]) + eps) / (4 + eps)
    q = (np.array([1, 3]) + eps) / (4 + eps)

    assert estimate_kl_divergence_categorical(x, x.copy()) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.sum(p * np.log(p / q)))

    # Distinct NaN objects, including NumPy floating NaNs, belong to the same joint category.
    a = np.array([["a", np.nan], ["b", "p"]], dtype=object)
    b = np.array([["a", float("nan")], ["b", "p"]], dtype=object)
    c = np.array([["a", np.float32("nan")], ["b", "p"]], dtype=object)
    assert estimate_kl_divergence_categorical(a, b) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(a, c) == pytest.approx(0)


def test_categorical_kl_does_not_merge_missing_values_with_other_categories():
    eps = np.finfo(np.float64).eps
    # Disjoint support: two categories have counts (2, 0) and (0, 2).
    expected = 2 / (2 + eps) * np.log((2 + eps) / eps)
    missing = np.array([np.nan, np.nan], dtype=object)
    string_nan = np.array(["nan", "nan"], dtype=object)
    none = np.array([None, None], dtype=object)

    assert estimate_kl_divergence_categorical(string_nan, missing) == pytest.approx(expected)
    assert estimate_kl_divergence_categorical(none, missing) == pytest.approx(expected)
    assert estimate_kl_divergence_categorical(np.array([0.0, 0.0]), np.array([-0.0, -0.0])) == pytest.approx(0)


@pytest.mark.parametrize(
    "make_missing",
    [
        lambda: complex("nan"),
        lambda: np.complex128(complex("nan")),
        lambda: np.datetime64("NaT", "D"),
        lambda: np.timedelta64("NaT", "s"),
        lambda: Decimal("NaN"),
    ],
    ids=["complex", "numpy-complex", "datetime64", "timedelta64", "decimal"],
)
def test_categorical_kl_treats_other_self_unequal_missing_values_as_one_category(make_missing):
    eps = np.finfo(np.float64).eps
    # Create a new object each time, so the categories cannot be merged by object identity.
    x = np.array([make_missing(), make_missing(), "a"], dtype=object)
    y = np.array([make_missing(), "a", "a"], dtype=object)
    # P(missing, a) = (2, 1)/3 and Q(missing, a) = (1, 2)/3.
    p = (np.array([2, 1]) + eps) / (3 + eps)
    q = (np.array([1, 2]) + eps) / (3 + eps)

    assert estimate_kl_divergence_categorical(x, x.copy()) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.sum(p * np.log(p / q)))


def test_categorical_kl_treats_missing_values_in_native_complex_and_datetime_arrays_as_one_category():
    complex_values = np.array([complex("nan"), 1, complex("nan")])
    dates = np.array(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]")

    assert estimate_kl_divergence_categorical(complex_values, complex_values.copy()) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(dates, dates.copy()) == pytest.approx(0)


def test_categorical_kl_keeps_values_with_non_boolean_self_comparison():
    eps = np.finfo(np.float64).eps
    expected = 2 / (2 + eps) * np.log((2 + eps) / eps)
    missing = np.array([np.nan, np.nan], dtype=object)
    infinite = np.array([np.inf, np.inf], dtype=object)

    class NonBooleanComparison:
        # Like pd.NA or array cells, comparing returns a non-boolean result.
        def __ne__(self, other):
            return np.array([True, False])

    value = NonBooleanComparison()
    non_boolean = np.array([value, value], dtype=object)

    assert estimate_kl_divergence_categorical(non_boolean, non_boolean.copy()) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(non_boolean, missing) == pytest.approx(expected)
    assert estimate_kl_divergence_categorical(infinite, missing) == pytest.approx(expected)


@pytest.mark.parametrize(
    "dtype, values, unit",
    [
        ("datetime64[D]", ["1500-01-01", "2500-01-01"], "s"),
        ("datetime64[Y]", ["1900", "2000"], "M"),
        ("datetime64[ps]", [1, 2], "as"),
        ("timedelta64[D]", [-1, 2], "s"),
        ("timedelta64[Y]", [-1, 2], "M"),
        ("timedelta64[ps]", [1, 2], "as"),
        ("timedelta64[2s]", [-1, 2], "ms"),
    ],
)
def test_categorical_kl_matches_equivalent_temporal_units(dtype, values, unit):
    x = np.array(values, dtype=dtype)
    y = x.astype(dtype.split("[")[0] + "[" + unit + "]")
    original_x, original_y = x.copy(), y.copy()

    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(x, y[::-1]) == pytest.approx(0)
    # Preserve distinct values even beyond the nanosecond date range or below nanosecond resolution.
    assert estimate_kl_divergence_categorical(x[:1], y[1:]) > 30
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
def test_categorical_kl_matches_temporal_units_in_mixed_joint_rows(kind):
    first = getattr(np, kind)(1, "D")
    second = getattr(np, kind)(2, "D")
    first_seconds = first.astype(kind + "[s]")
    second_seconds = second.astype(kind + "[s]")
    x = np.array([["a", first], ["a", first_seconds], ["b", second]], dtype=object)
    y = np.array([["a", first_seconds], ["b", second_seconds], ["b", second]], dtype=object)

    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.log(2) / 3)
    assert estimate_kl_divergence_categorical(x, x[::-1].copy()) == pytest.approx(0)


def test_categorical_kl_supports_orderable_unhashable_categories():
    from dataclasses import dataclass

    @dataclass(order=True)
    class Category:
        value: int

    x = np.array([Category(1), Category(1), Category(2)], dtype=object)
    y = np.array([Category(1), Category(2), Category(2)], dtype=object)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.log(2) / 3)
    assert estimate_kl_divergence_categorical(x, x[::-1].copy()) == pytest.approx(0)

    # Unhashable cells can occur in joint observations alongside ordinary and missing categories.
    x = np.array([["a", Category(1), Decimal("NaN")]] * 2 + [["b", Category(2), None]], dtype=object)
    y = np.array([["a", Category(1), float("nan")]] + [["b", Category(2), None]] * 2, dtype=object)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.log(2) / 3)
    assert estimate_kl_divergence_categorical(x, y[:, ::-1]) > 30


def test_categorical_kl_keeps_non_boolean_comparisons_in_temporal_fallback():
    class NonBooleanComparison:
        def __eq__(self, other):
            return np.array([True, False])

        def __ne__(self, other):
            return np.array([False, True])

        __hash__ = object.__hash__

    first, second = NonBooleanComparison(), NonBooleanComparison()
    x = np.array([np.datetime64("2020-01-01"), first, second, np.nan, None, "nan", True, 1], dtype=object)
    y = x[::-1].copy()
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_categorical_kl_supports_mixed_decimal_and_timedelta_categories(reverse, joint):
    x = np.array([np.timedelta64(1, "D"), np.timedelta64(1, "D"), Decimal("0.1")], dtype=object)
    y = np.array([np.timedelta64(1, "D"), Decimal("0.1"), Decimal("0.1")], dtype=object)
    if reverse:
        x, y = x[::-1].copy(), y[::-1].copy()
    if joint:
        x = np.column_stack([np.full(len(x), "same", dtype=object), x])
        y = np.column_stack([np.full(len(y), "same", dtype=object), y])

    assert estimate_kl_divergence_categorical(x, x.copy()) == pytest.approx(0)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.log(2) / 3)


@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
@pytest.mark.parametrize("joint", [False, True])
def test_categorical_kl_groups_ordered_temporal_categories_without_scanning_every_pair(monkeypatch, kind, joint):
    import dowhy.gcm.divergence as divergence

    size = 256
    x = np.arange(size).astype(kind + "[D]")
    x[0] = "NaT"
    y = x.astype(kind + "[s]")[::-1].copy()
    if joint:
        # Construct object rows without converting NumPy temporal scalars to Python dates or timedeltas.
        x = np.array([["same", value] for value in x], dtype=object)
        y = np.array([["same", value] for value in y], dtype=object)

    comparisons = 0
    original_equal = divergence._joint_categories_equal

    def count_comparisons(left, right):
        nonlocal comparisons
        comparisons += 1
        return original_equal(left, right)

    monkeypatch.setattr(divergence, "_joint_categories_equal", count_comparisons)
    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(0)
    # Bound category-search work deterministically, independently of machine speed.
    assert comparisons < 20 * size


def test_categorical_kl_preserves_first_seen_equality_for_mixed_durations_and_integers():
    # NumPy can compare an integer equal to durations in different units, even when those durations are unequal.
    # Keep the original first-seen representative for these genuinely mixed categories.
    x = np.array([np.timedelta64(1, "s"), 2, 1, np.timedelta64(1, "D"), np.timedelta64(1, "s")], dtype=object)
    y = np.array([2, np.timedelta64(1, "D"), 1, 2, np.timedelta64(1, "s")], dtype=object)
    eps = np.finfo(float).eps
    p = (np.array([3, 1, 1]) + eps) / (5 + eps)
    q = (np.array([2, 2, 1]) + eps) / (5 + eps)

    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.sum(p * np.log(p / q)))


def test_categorical_kl_falls_back_for_partially_ordered_unhashable_categories():
    # Disjoint sets have neither less-than nor greater-than ordering, so sorting cannot group their duplicates.
    x = np.array([{1}, {2}, {1}], dtype=object)
    y = np.array([{1}, {2}, {2}], dtype=object)

    assert estimate_kl_divergence_categorical(x, y) == pytest.approx(np.log(2) / 3)
    assert estimate_kl_divergence_categorical(x, x[::-1].copy()) == pytest.approx(0)
