import numpy as np
import pytest

from dowhy.gcm import confidence_intervals
from dowhy.gcm.confidence_intervals import estimate_geometric_median


def test_given_simple_counter_as_estimation_func_when_confidence_interval_then_returns_mean_and_interval_counter():
    i = 0.0

    def simple_counter():
        nonlocal i
        i += 1.0
        return {"X": i}

    median, interval = confidence_intervals(simple_counter, num_bootstrap_resamples=20)

    assert median["X"] == pytest.approx(10.5)
    # Default confidence_level=0.95 -> two-sided 2.5th/97.5th percentiles of the results 1..20.
    assert np.allclose(interval["X"], [1.475, 19.525])


# ---------------------------------------------------------------------------
# confidence_intervals – numpy array output
# ---------------------------------------------------------------------------


def test_given_array_returning_func_when_confidence_interval_then_returns_array_median_and_2d_ci():
    """confidence_intervals should handle functions that return a numpy array."""
    i = [0]

    def counter():
        i[0] += 1
        return np.array([float(i[0])])

    median, interval = confidence_intervals(counter, num_bootstrap_resamples=20)

    assert median.shape == (1,)
    assert interval.shape == (1, 2)
    assert median[0] == pytest.approx(10.5)
    assert np.allclose(interval[0], [1.475, 19.525])


def test_given_multidim_array_returning_func_when_confidence_interval_then_ci_has_correct_shape():
    """confidence_intervals should return a CI row for each output dimension."""
    i = [0]

    def counter():
        i[0] += 1
        v = float(i[0])
        return np.array([v, v * 2])

    median, interval = confidence_intervals(counter, num_bootstrap_resamples=20)

    assert median.shape == (2,)
    assert interval.shape == (2, 2)
    # Second dimension is exactly twice the first.
    assert median[1] == pytest.approx(2 * median[0])
    assert np.allclose(interval[1], 2 * interval[0])


# ---------------------------------------------------------------------------
# confidence_intervals – dict output with multiple keys
# ---------------------------------------------------------------------------


def test_given_dict_func_with_multiple_keys_when_confidence_interval_then_returns_dict_for_each_key():
    i = [0]

    def counter():
        i[0] += 1
        v = float(i[0])
        return {"A": v, "B": v * 3}

    median, interval = confidence_intervals(counter, num_bootstrap_resamples=20)

    assert set(median.keys()) == {"A", "B"}
    assert set(interval.keys()) == {"A", "B"}
    assert median["A"] == pytest.approx(10.5)
    assert median["B"] == pytest.approx(31.5)
    assert np.allclose(interval["A"], [1.475, 19.525])
    assert np.allclose(interval["B"], [3 * 1.475, 3 * 19.525])


# ---------------------------------------------------------------------------
# confidence_intervals – confidence_level parameter
# ---------------------------------------------------------------------------


def test_given_narrow_confidence_level_when_confidence_interval_then_interval_is_smaller():
    """A tighter confidence_level produces a narrower interval than the default 0.95."""
    i = [0]

    def counter():
        i[0] += 1
        return np.array([float(i[0])])

    _, interval_wide = confidence_intervals(counter, num_bootstrap_resamples=20, confidence_level=0.95)
    i[0] = 0
    _, interval_narrow = confidence_intervals(counter, num_bootstrap_resamples=20, confidence_level=0.50)

    wide_width = interval_wide[0, 1] - interval_wide[0, 0]
    narrow_width = interval_narrow[0, 1] - interval_narrow[0, 0]
    assert narrow_width < wide_width


# ---------------------------------------------------------------------------
# confidence_intervals – custom summary function
# ---------------------------------------------------------------------------


def test_given_custom_summary_func_when_confidence_interval_then_custom_func_is_applied():
    """confidence_intervals should use the caller-supplied summary function."""
    i = [0]

    def counter():
        i[0] += 1
        return np.array([float(i[0])])

    # Use np.mean (column-wise) instead of geometric median.
    def mean_summary(results):
        return np.mean(results, axis=0)

    median, _ = confidence_intervals(counter, num_bootstrap_resamples=20, bootstrap_results_summary_func=mean_summary)

    assert median[0] == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# confidence_intervals – parallel execution
# ---------------------------------------------------------------------------


def test_given_n_jobs_2_when_confidence_interval_then_returns_same_result_structure():
    """Parallel n_jobs should still return results with the correct structure."""
    i = [0]

    def counter():
        i[0] += 1
        return np.array([float(i[0])])

    median, interval = confidence_intervals(counter, num_bootstrap_resamples=10, n_jobs=2)

    assert median.shape == (1,)
    assert interval.shape == (1, 2)
    assert interval[0, 0] <= median[0] <= interval[0, 1]


# ---------------------------------------------------------------------------
# confidence_intervals – error conditions
# ---------------------------------------------------------------------------


def test_given_zero_resamples_when_confidence_interval_then_raises_value_error():
    with pytest.raises(ValueError, match="greater than 0"):
        confidence_intervals(lambda: np.array([1.0]), num_bootstrap_resamples=0)


def test_given_negative_resamples_when_confidence_interval_then_raises_value_error():
    with pytest.raises(ValueError, match="greater than 0"):
        confidence_intervals(lambda: np.array([1.0]), num_bootstrap_resamples=-5)


# ---------------------------------------------------------------------------
# estimate_geometric_median
# ---------------------------------------------------------------------------


def test_given_points_symmetric_around_origin_when_estimate_geometric_median_then_returns_near_zero():
    """Symmetric point sets have geometric median near the centre."""
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    gm = estimate_geometric_median(X)
    assert gm == pytest.approx(np.zeros(2), abs=1e-3)


def test_given_collinear_points_when_estimate_geometric_median_then_returns_midpoint():
    """For three collinear points, the geometric median is the middle point."""
    X = np.array([[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    gm = estimate_geometric_median(X)
    assert gm == pytest.approx(np.array([0.0, 0.0]), abs=1e-3)


def test_given_cluster_with_distant_outlier_when_estimate_geometric_median_then_robust_to_outlier():
    """The geometric median is more robust to a single extreme outlier than the mean."""
    rng = np.random.default_rng(42)
    cluster = rng.normal(loc=0.0, scale=0.1, size=(99, 2))
    outlier = np.array([[1000.0, 1000.0]])
    X = np.vstack([cluster, outlier])

    gm = estimate_geometric_median(X)
    mean = np.mean(X, axis=0)

    # The geometric median stays close to the cluster; the mean does not.
    assert np.linalg.norm(gm) < np.linalg.norm(mean)
