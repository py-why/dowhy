"""Unit tests for do_samplers module."""

import logging

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dowhy import EstimandType
from dowhy.do_samplers.kernel_density_sampler import KernelDensitySampler
from dowhy.do_samplers.mcmc_sampler import MCMCSampler
from dowhy.do_samplers.multivariate_weighting_sampler import MultivariateWeightingSampler
from dowhy.do_samplers.weighting_sampler import WeightingSampler


@pytest.fixture
def simple_graph():
    """Create a simple causal graph: Z -> T -> Y."""
    g = nx.DiGraph()
    g.add_edges_from([("Z", "T"), ("T", "Y")])
    return g


@pytest.fixture
def simple_data():
    """Generate simple test data matching the simple graph."""
    np.random.seed(42)
    data = pd.DataFrame({
        "Z": np.random.normal(0, 1, 100),
        "T": np.random.normal(0, 1, 100),
        "Y": np.random.normal(0, 1, 100),
    })
    return data


@pytest.fixture
def confounded_graph():
    """Create a confounded causal graph: Z -> T, Z -> Y, T -> Y."""
    g = nx.DiGraph()
    g.add_edges_from([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    return g


@pytest.fixture
def confounded_data():
    """Generate confounded data where Z affects both T and Y."""
    np.random.seed(42)
    n = 200
    z = np.random.normal(0, 1, n)
    t = z + np.random.normal(0, 0.5, n)
    y = z + t + np.random.normal(0, 0.5, n)
    data = pd.DataFrame({"Z": z, "T": t, "Y": y})
    return data


class TestWeightingSampler:
    """Tests for WeightingSampler class."""

    def test_weighting_sampler_initialization(self, simple_graph, simple_data):
        """Test WeightingSampler can be initialized."""
        sampler = WeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler is not None
        assert sampler.point_sampler is False

    def test_weighting_sampler_init_preserves_data(self, simple_graph, simple_data):
        """Test WeightingSampler preserves original data."""
        sampler = WeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler._data.shape == simple_data.shape
        pd.testing.assert_frame_equal(sampler._data, simple_data)

    def test_weighting_sampler_keep_original_treatment(self, simple_graph, simple_data):
        """Test keep_original_treatment parameter."""
        sampler = WeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            keep_original_treatment=True,
        )
        assert sampler.keep_original_treatment is True

    def test_weighting_sampler_compute_weights(self, confounded_data, confounded_graph):
        """Test weight computation in weighting sampler."""
        sampler = WeightingSampler(
            graph=confounded_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=confounded_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        # Add propensity scores manually
        sampler._df["propensity_score"] = np.abs(np.random.normal(0.5, 0.1, len(confounded_data)))
        weights = sampler.compute_weights()
        assert len(weights) == len(confounded_data)
        assert np.all(weights > 0), "All weights must be positive"

    def test_weighting_sampler_discrete_treatment_no_match(self, simple_graph, simple_data):
        """Test error when discrete treatment value doesn't match observed data."""
        sampler = WeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        # Try to set treatment to value not in data
        with pytest.raises(ValueError, match="do not exactly match any observed"):
            sampler.make_treatment_effective({"T": 999})


class TestKernelDensitySampler:
    """Tests for KernelDensitySampler class."""

    def test_kernel_density_sampler_initialization(self, simple_graph, simple_data):
        """Test KernelDensitySampler can be initialized."""
        sampler = KernelDensitySampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler is not None

    def test_kernel_density_sampler_init_preserves_data(self, simple_graph, simple_data):
        """Test KernelDensitySampler preserves original data."""
        sampler = KernelDensitySampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        pd.testing.assert_frame_equal(sampler._data, simple_data)

    def test_kernel_density_sampler_treatment_assignment(self, simple_graph, simple_data):
        """Test treatment assignment in KernelDensitySampler."""
        sampler = KernelDensitySampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            keep_original_treatment=False,
        )
        treatment_value = {"T": 0.5}
        sampler.make_treatment_effective(treatment_value)
        np.testing.assert_array_almost_equal(sampler._df["T"].values, 0.5)

    def test_kernel_density_sampler_reset(self, simple_graph, simple_data):
        """Test reset method restores original data."""
        sampler = KernelDensitySampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        original_df = sampler._df.copy()
        sampler._df["T"] = 999
        sampler.reset()
        pd.testing.assert_frame_equal(sampler._df, original_df)


class TestMCMCSampler:
    """Tests for MCMCSampler class."""

    def test_mcmc_sampler_initialization(self, simple_graph, simple_data):
        """Test MCMCSampler can be initialized."""
        sampler = MCMCSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler is not None
        assert hasattr(sampler, "sampler")

    def test_mcmc_sampler_init_preserves_data(self, simple_graph, simple_data):
        """Test MCMCSampler preserves original data."""
        sampler = MCMCSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        pd.testing.assert_frame_equal(sampler._data, simple_data)

    def test_mcmc_sampler_treatment_assignment(self, simple_graph, simple_data):
        """Test treatment assignment in MCMCSampler."""
        sampler = MCMCSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            keep_original_treatment=False,
        )
        treatment_value = {"T": 1.0}
        sampler.make_treatment_effective(treatment_value)
        np.testing.assert_array_almost_equal(sampler._df["T"].values, 1.0)

    def test_mcmc_sampler_point_sampler_flag(self, simple_graph, simple_data):
        """Test point_sampler flag indicates point-wise sampling."""
        sampler = MCMCSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler.point_sampler is True


class TestMultivariateWeightingSampler:
    """Tests for MultivariateWeightingSampler class."""

    def test_multivariate_weighting_sampler_initialization(self, simple_graph, simple_data):
        """Test MultivariateWeightingSampler can be initialized."""
        sampler = MultivariateWeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler is not None
        assert sampler.point_sampler is False

    def test_multivariate_weighting_sampler_init_preserves_data(self, simple_graph, simple_data):
        """Test MultivariateWeightingSampler preserves original data."""
        sampler = MultivariateWeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        pd.testing.assert_frame_equal(sampler._data, simple_data)

    def test_multivariate_weighting_sampler_treatment_assignment(self, simple_graph, simple_data):
        """Test treatment assignment in MultivariateWeightingSampler."""
        sampler = MultivariateWeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            keep_original_treatment=False,
        )
        treatment_value = {"T": 0.25}
        sampler.make_treatment_effective(treatment_value)
        np.testing.assert_array_almost_equal(sampler._df["T"].values, 0.25)

    def test_multivariate_weighting_sampler_multiple_treatments(self):
        """Test MultivariateWeightingSampler with multiple treatment variables."""
        g = nx.DiGraph()
        g.add_edges_from([("Z", "T1"), ("Z", "T2"), ("T1", "Y"), ("T2", "Y")])
        data = pd.DataFrame({
            "Z": np.random.normal(0, 1, 100),
            "T1": np.random.normal(0, 1, 100),
            "T2": np.random.normal(0, 1, 100),
            "Y": np.random.normal(0, 1, 100),
        })
        sampler = MultivariateWeightingSampler(
            graph=g,
            action_nodes=["T1", "T2"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T1", "T2", "Y"],
            data=data,
            variable_types={"Z": "c", "T1": "c", "T2": "c", "Y": "c"},
            keep_original_treatment=False,
        )
        treatment_value = {"T1": 0.5, "T2": -0.5}
        sampler.make_treatment_effective(treatment_value)
        np.testing.assert_array_almost_equal(sampler._df["T1"].values, 0.5)
        np.testing.assert_array_almost_equal(sampler._df["T2"].values, -0.5)


class TestDoSamplerCommonBehavior:
    """Tests for common behavior across all samplers."""

    @pytest.mark.parametrize("sampler_class", [
        WeightingSampler,
        KernelDensitySampler,
        MCMCSampler,
        MultivariateWeightingSampler,
    ])
    def test_samplers_reset_method(self, simple_graph, simple_data, sampler_class):
        """Test reset method works for all samplers."""
        sampler = sampler_class(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        original_shape = sampler._df.shape
        sampler._df.iloc[0, 0] = 999
        sampler.reset()
        assert sampler._df.shape == original_shape

    @pytest.mark.parametrize("sampler_class", [
        WeightingSampler,
        KernelDensitySampler,
        MCMCSampler,
        MultivariateWeightingSampler,
    ])
    def test_samplers_init_with_params(self, simple_graph, simple_data, sampler_class):
        """Test samplers accept additional params."""
        params = {"custom_param": 42}
        sampler = sampler_class(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            params=params,
        )
        assert hasattr(sampler, "custom_param")
        assert sampler.custom_param == 42

    @pytest.mark.parametrize("sampler_class", [
        WeightingSampler,
        KernelDensitySampler,
        MCMCSampler,
        MultivariateWeightingSampler,
    ])
    def test_samplers_init_with_num_cores(self, simple_graph, simple_data, sampler_class):
        """Test samplers accept num_cores parameter."""
        sampler = sampler_class(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
            num_cores=2,
        )
        assert sampler.num_cores == 2

    @pytest.mark.parametrize("sampler_class", [
        WeightingSampler,
        KernelDensitySampler,
        MCMCSampler,
        MultivariateWeightingSampler,
    ])
    def test_samplers_treatment_names_parsed(self, simple_graph, simple_data, sampler_class):
        """Test treatment names are properly parsed."""
        sampler = sampler_class(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler._treatment_names == ["T"]

    @pytest.mark.parametrize("sampler_class", [
        WeightingSampler,
        KernelDensitySampler,
        MCMCSampler,
        MultivariateWeightingSampler,
    ])
    def test_samplers_outcome_names_parsed(self, simple_graph, simple_data, sampler_class):
        """Test outcome names are properly parsed."""
        sampler = sampler_class(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        assert sampler._outcome_names == ["Y"]


class TestDoSamplerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_sampler_with_empty_data(self, simple_graph):
        """Test sampler with empty DataFrame."""
        empty_data = pd.DataFrame(columns=["Z", "T", "Y"])
        with pytest.raises((ValueError, KeyError)):
            WeightingSampler(
                graph=simple_graph,
                action_nodes=["T"],
                outcome_nodes=["Y"],
                observed_nodes=["Z", "T", "Y"],
                data=empty_data,
                variable_types={"Z": "c", "T": "c", "Y": "c"},
            )

    def test_sampler_with_missing_variable_types(self, simple_graph, simple_data):
        """Test sampler requires variable_types."""
        with pytest.raises(NotImplementedError, match="Variable type inference not implemented"):
            WeightingSampler(
                graph=simple_graph,
                action_nodes=["T"],
                outcome_nodes=["Y"],
                observed_nodes=["Z", "T", "Y"],
                data=simple_data,
            )

    def test_sampler_with_nan_values(self, simple_graph):
        """Test sampler behavior with NaN values in data."""
        data_with_nan = pd.DataFrame({
            "Z": [1.0, np.nan, 3.0],
            "T": [1.0, 2.0, 3.0],
            "Y": [1.0, 2.0, 3.0],
        })
        # Sampler should handle or raise informative error
        with pytest.raises((ValueError, KeyError)):
            WeightingSampler(
                graph=simple_graph,
                action_nodes=["T"],
                outcome_nodes=["Y"],
                observed_nodes=["Z", "T", "Y"],
                data=data_with_nan,
                variable_types={"Z": "c", "T": "c", "Y": "c"},
            )

    def test_sampler_outcome_support_computation(self, simple_graph, simple_data):
        """Test outcome support bounds are computed correctly."""
        sampler = WeightingSampler(
            graph=simple_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["Z", "T", "Y"],
            data=simple_data,
            variable_types={"Z": "c", "T": "c", "Y": "c"},
        )
        expected_lower = simple_data["Y"].min()
        expected_upper = simple_data["Y"].max()
        assert sampler.outcome_lower_support[0] == expected_lower
        assert sampler.outcome_upper_support[0] == expected_upper
