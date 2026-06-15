import numpy as np
import pandas as pd

from dowhy import CausalModel


def _make_data(n=400):
    rng = np.random.RandomState(0)
    w = rng.normal(size=n)
    v = (rng.uniform(size=n) < 1 / (1 + np.exp(-w))).astype(int)
    y = 2 * v + w + rng.normal(size=n)
    return pd.DataFrame({"v0": v, "W0": w, "y": y})


def _estimate(random_state):
    data = _make_data()
    model = CausalModel(data=data, treatment="v0", outcome="y", common_causes=["W0"])
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        test_significance=True,
        confidence_intervals=True,
        method_params={
            "init_params": {"random_state": random_state},
            "num_null_simulations": 20,
            "num_simulations": 50,
        },
    )


def test_bootstrap_confidence_intervals_are_reproducible_with_random_state():
    first = _estimate(random_state=42).get_confidence_intervals()
    second = _estimate(random_state=42).get_confidence_intervals()
    assert np.allclose(first, second)


def test_bootstrap_significance_is_reproducible_with_random_state():
    first = _estimate(random_state=42).test_stat_significance()["p_value"]
    second = _estimate(random_state=42).test_stat_significance()["p_value"]
    assert np.allclose(first, second)


def test_bootstrap_confidence_intervals_differ_across_random_states():
    first = _estimate(random_state=42).get_confidence_intervals()
    second = _estimate(random_state=7).get_confidence_intervals()
    assert not np.allclose(first, second)
