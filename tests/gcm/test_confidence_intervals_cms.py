import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ProbabilisticCausalModel,
    bootstrap_sampling,
    draw_samples,
    fit_and_compute,
)
from dowhy.gcm.confidence_intervals import confidence_intervals
from dowhy.gcm.ml import create_hist_gradient_boost_regressor


@flaky(max_runs=2)
def test_given_causal_graph_based_estimation_func_when_confidence_interval_then_can_use_fit_and_compute():
    def draw_single_sample(causal_graph, variable):
        return draw_samples(causal_graph, 1)[variable][0]

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X", "Y")]))
    causal_model.set_causal_mechanism("X", EmpiricalDistribution())
    causal_model.set_causal_mechanism("Y", AdditiveNoiseModel(create_hist_gradient_boost_regressor()))

    median, interval = confidence_intervals(
        fit_and_compute(
            draw_single_sample,
            causal_model,
            bootstrap_training_data=pd.DataFrame(
                {"X": [1, 3, 4, 4, 1, 3, 6, 3, 3, 6, 3, 4], "Y": [4, 5, 4, 4, 4, 5, 3, 6, 5, 3, 6, 4]}
            ),
            bootstrap_data_subset_size_fraction=0.5,
            variable="X",
        )
    )

    assert median == pytest.approx(4.0, abs=1.1)
    assert np.allclose(interval, [1.0, 6.0], atol=2.0)


def test_given_parameterized_estimation_func_when_confidence_interval_then_can_use_bootstrap_sampling_to_bind_parameters():
    i = 0.0

    def parameterized_counter(some_parameter):
        nonlocal i
        i += 1.0
        return {some_parameter: i}

    median, interval = confidence_intervals(bootstrap_sampling(parameterized_counter, "A"), num_bootstrap_resamples=20)

    assert median["A"] == pytest.approx(10.5)
    assert np.allclose(interval["A"], [1.95, 19.05])
