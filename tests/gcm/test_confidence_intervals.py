import numpy as np
import pytest

from dowhy.gcm import confidence_intervals


def test_given_simple_counter_as_estimation_func_when_confidence_interval_then_returns_mean_and_interval_counter():
    i = 0.0

    def simple_counter():
        nonlocal i
        i += 1.0
        return {"X": i}

    median, interval = confidence_intervals(simple_counter, num_bootstrap_resamples=20)

    assert median["X"] == pytest.approx(10.5)
    assert np.allclose(interval["X"], [1.95, 19.05])
