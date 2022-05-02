import numpy as np
import pytest
from flaky import flaky
from pytest import approx

from dowhy.gcm.stats import quantile_based_fwer
from dowhy.gcm.util.general import geometric_median


@flaky(max_runs=5)
def test_estimate_geometric_median():
    a = np.random.normal(10, 1, 100)
    a = np.hstack([a, np.random.normal(10000, 1, 20)])
    b = np.random.normal(-5, 1, 100)
    b = np.hstack([b, np.random.normal(-10000, 1, 20)])

    gm = geometric_median(np.vstack([a, b]).T)

    assert gm[0] == approx(10, abs=0.5)
    assert gm[1] == approx(-5, abs=0.5)


def test_quantile_based_fwer():
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    assert quantile_based_fwer(p_values, quantile=0.5) == 0.055 / 0.5
    assert quantile_based_fwer(p_values, quantile=0.25) == 0.0325 / 0.25
    assert quantile_based_fwer(p_values, quantile=0.75) == 0.0775 / 0.75

    assert quantile_based_fwer(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1]),
                               quantile=0.5) == 0.06 / 0.5
    assert quantile_based_fwer(np.array([0.9, 0.95, 1]), quantile=0.5) == 1
    assert quantile_based_fwer(np.array([0, 0, 0]), quantile=0.5) == 0
    assert quantile_based_fwer(np.array([0.33]), quantile=0.5) == 0.33


def test_given_p_values_with_nans_when_using_quantile_based_fwer_then_ignores_the_nan_values():
    p_values = np.array([0.01, np.nan, 0.02, 0.03, 0.04, 0.05, np.nan, 0.06, 0.07, 0.08, 0.09, 0.1])
    assert quantile_based_fwer(p_values, quantile=0.5) == 0.055 / 0.5


def test_quantile_based_fwer_scaling():
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    p_values_scaling = np.array([2, 2, 1, 2, 1, 3, 1, 2, 4, 1])

    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.5) == approx(0.15)
    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.25) == approx(0.17)
    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.75) == approx(0.193, abs=0.001)


def test_quantile_based_fwer_raises_error():
    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=0)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), np.array([1, 2]), quantile=0.1)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=1.1)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=-0.5)
