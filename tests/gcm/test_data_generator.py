import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel
from dowhy.gcm.causal_models import StructuralCausalModel
from dowhy.gcm.data_generator import (
    DataGeneratorConfig,
    _NonAdditiveNoiseFCM,
    _TransformedConditionalModel,
    assign_random_fcms,
    generate_random_dag,
    generate_random_scm,
    generate_samples_from_random_scm,
)
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node


def test_generate_random_dag_node_counts():
    dag = generate_random_dag(3, 5)
    assert dag.number_of_nodes() == 8
    assert nx.is_directed_acyclic_graph(dag)


def test_generate_random_dag_sparse():
    dag = generate_random_dag(5, 10, edge_density=0.0)
    assert dag.number_of_edges() >= 10  # at least 1 parent per child
    assert nx.is_weakly_connected(dag)
    assert nx.is_directed_acyclic_graph(dag)


def test_generate_random_dag_dense():
    dag = generate_random_dag(5, 10, edge_density=1.0)
    max_edges = sum(range(5, 15))  # child i connects to all (5+i) predecessors
    assert dag.number_of_edges() == max_edges


def test_generate_random_scm_returns_structural_causal_model():
    np.random.seed(0)
    scm = generate_random_scm(3, 4)
    assert isinstance(scm, StructuralCausalModel)
    assert scm.graph.number_of_nodes() == 7


def test_generate_random_dag_with_zero_roots_does_not_raise():
    # Regression: with num_roots=0 the first child had no candidate parents, so np.random.choice on an empty node list
    # raised. It should instead become a root node.
    np.random.seed(0)
    dag = generate_random_dag(0, 3)
    assert dag.number_of_nodes() == 3
    assert nx.is_directed_acyclic_graph(dag)


def test_unimodal_uniform_noise_has_requested_standard_deviation():
    # Regression: the uniform noise branch used loc=-std, scale=2*std, giving an actual standard deviation of
    # std/sqrt(3). It must honor noise_std_range as the standard deviation, matching the Gaussian branch.
    from dowhy.gcm.data_generator import _create_noise_model

    np.random.seed(0)
    cfg = DataGeneratorConfig(noise_std_range=(0.15, 0.15), prob_unimodal_noise=1.0)
    empirical_stds = [_create_noise_model(cfg).draw_samples(20000).std() for _ in range(40)]
    # Averaging over both the Gaussian and uniform branches, the empirical std should track the requested 0.15.
    assert np.mean(empirical_stds) == pytest.approx(0.15, abs=0.02)


def test_generate_random_scm_values_are_bounded():
    np.random.seed(0)
    scm = generate_random_scm(5, 10)

    samples = draw_samples(scm, 2000)
    assert samples.shape == (2000, 15)
    assert samples.min().min() > -20
    assert samples.max().max() < 20


def test_generate_samples_from_random_scm_shape():
    np.random.seed(0)
    samples = generate_samples_from_random_scm(3, 4, 500)
    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (500, 7)


def test_all_linear_config():
    cfg = DataGeneratorConfig(
        prob_linear_mechanism=1.0,
        prob_non_additive_noise=0.0,
        prob_heteroscedastic_noise=0.0,
        prob_log_space_mechanism=0.0,
    )
    np.random.seed(0)
    scm = generate_random_scm(2, 5, cfg)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            m = scm.causal_mechanism(node)
            base = m._base if isinstance(m, _TransformedConditionalModel) else m
            assert isinstance(base, AdditiveNoiseModel)
            assert len(base.prediction_model._weights) == 1  # no hidden layer: a calibrated linear function


def test_all_non_additive_config():
    cfg = DataGeneratorConfig(prob_non_additive_noise=1.0, prob_linear_mechanism=0.0)
    np.random.seed(0)
    scm = generate_random_scm(2, 5, cfg)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            m = scm.causal_mechanism(node)
            base = m._base if isinstance(m, _TransformedConditionalModel) else m
            assert isinstance(base, _NonAdditiveNoiseFCM)


def test_clipped_positive_root_nodes():
    cfg = DataGeneratorConfig(prob_clipped_positive=1.0, prob_clipped_negative=0.0, prob_discretised=0.0)
    np.random.seed(0)
    scm = generate_random_scm(3, 3, cfg)

    samples = draw_samples(scm, 1000)
    for col in samples.columns:
        assert samples[col].min() >= -1e-9


def test_discretised_root_nodes():
    cfg = DataGeneratorConfig(
        prob_discretised=1.0, discrete_num_bins_range=(3, 3), prob_clipped_positive=0.0, prob_clipped_negative=0.0
    )
    np.random.seed(0)
    scm = generate_random_scm(2, 3, cfg)

    samples = draw_samples(scm, 1000)
    for col in samples.columns:
        assert samples[col].nunique() <= 4


def test_assign_random_fcms_on_existing_graph():
    graph = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "C")])
    scm = assign_random_fcms(graph)
    assert isinstance(scm, StructuralCausalModel)
    assert set(scm.graph.nodes) == {"A", "B", "C"}

    samples = draw_samples(scm, 100)
    assert samples.shape == (100, 3)


def test_reproducibility_with_seed():
    np.random.seed(123)
    s1 = generate_samples_from_random_scm(3, 3, 100)
    np.random.seed(123)
    s2 = generate_samples_from_random_scm(3, 3, 100)
    pd.testing.assert_frame_equal(s1, s2)


def test_generate_random_dag_without_children_keeps_all_roots():
    # Regression: the weak-connectivity merge ran for num_children == 0 and chained all roots together.
    dag = generate_random_dag(3, 0)
    assert dag.number_of_nodes() == 3
    assert dag.number_of_edges() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(edge_density=1.5),
        dict(prob_linear_mechanism=-0.1),
        dict(prob_clipped_positive=0.7, prob_clipped_negative=0.5),
        dict(nn_hidden_units_range=(0, 4)),
        dict(nn_hidden_layers_range=(3, 1)),
        dict(noise_std_range=(0.0, 0.1)),
        dict(discrete_num_bins_range=(1, 3)),
        dict(noise_mixture_component_std=0.0),
        dict(nn_output_value_range=(1.0, -1.0)),
        dict(num_calibration_samples=1),
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        DataGeneratorConfig(**kwargs)


def test_default_config_is_valid():
    DataGeneratorConfig()


def test_mixture_noise_has_requested_standard_deviation():
    # Regression: only the component stds were rescaled, so mixture noise had std ~0.5 regardless of noise_std_range.
    from dowhy.gcm.data_generator import _create_noise_model

    np.random.seed(0)
    cfg = DataGeneratorConfig(noise_std_range=(0.15, 0.15), prob_unimodal_noise=0.0)
    stds = [_create_noise_model(cfg).draw_samples(20000).std() for _ in range(40)]
    assert max(abs(s - 0.15) for s in stds) < 0.02


def test_random_mixture_is_standardised_and_multimodal():
    from scipy import stats

    from dowhy.gcm.data_generator import _random_mixture

    np.random.seed(0)
    cfg = DataGeneratorConfig()
    n_multimodal = 0
    grid = np.linspace(-4, 4, 801)
    for _ in range(200):
        mixture = _random_mixture(cfg)
        x = mixture.draw_samples(20000).squeeze()
        assert abs(x.mean()) < 0.05
        assert abs(x.std() - 1.0) < 0.05
        density = sum(stats.norm.pdf(grid, mu, sd) for mu, sd in zip(mixture._means, mixture._stds))
        n_modes = np.sum((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]))
        n_multimodal += int(n_modes >= 2)
    assert n_multimodal >= 140  # ~90% expected with mean range 3.0 and component std 0.4


def test_log_uniform_noise_std_spreads_over_decades():
    from dowhy.gcm.data_generator import _draw_noise_std

    np.random.seed(0)
    cfg = DataGeneratorConfig(noise_std_range=(0.01, 1.0), noise_std_log_uniform=True)
    stds = np.array([_draw_noise_std(cfg) for _ in range(5000)])
    assert np.all((stds >= 0.01) & (stds <= 1.0))
    assert abs(np.median(stds) - 0.1) < 0.02  # geometric midpoint of the range


def test_noise_std_is_uniform_by_default():
    from dowhy.gcm.data_generator import _draw_noise_std

    np.random.seed(0)
    cfg = DataGeneratorConfig(noise_std_range=(0.01, 1.0))
    stds = np.array([_draw_noise_std(cfg) for _ in range(5000)])
    assert abs(np.median(stds) - 0.505) < 0.03


def test_mechanism_type_probabilities_are_marginal():
    # Regression: two sequential draws gave P(linear) = (1 - p_na) * p_lin instead of p_lin.
    from dowhy.gcm.data_generator import _create_non_root_model

    np.random.seed(0)
    cfg = DataGeneratorConfig(
        prob_non_additive_noise=0.3,
        prob_linear_mechanism=0.5,
        prob_heteroscedastic_noise=0.0,
        prob_log_space_mechanism=0.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
    )
    counts = {"non_additive": 0, "linear": 0, "nn": 0}
    for _ in range(3000):
        mechanism = _create_non_root_model(2, cfg)
        if isinstance(mechanism, _NonAdditiveNoiseFCM):
            counts["non_additive"] += 1
        elif len(mechanism.prediction_model._weights) == 1:
            counts["linear"] += 1
        else:
            counts["nn"] += 1
    assert counts["non_additive"] / 3000 == pytest.approx(0.3, abs=0.03)
    assert counts["linear"] / 3000 == pytest.approx(0.5, abs=0.03)
    assert counts["nn"] / 3000 == pytest.approx(0.2, abs=0.03)


def test_mechanism_probabilities_must_sum_to_at_most_one():
    with pytest.raises(ValueError):
        DataGeneratorConfig(prob_non_additive_noise=0.7, prob_linear_mechanism=0.5)


def test_clone_reproduces_generated_scm():
    # Regression: NN clones lost their calibration and linear clones raised NotFittedError.
    cfg = DataGeneratorConfig(prob_linear_mechanism=0.5, prob_non_additive_noise=0.3)
    np.random.seed(0)
    scm = generate_random_scm(3, 5, cfg)
    np.random.seed(1)
    original = draw_samples(scm, 500)
    np.random.seed(1)
    cloned = draw_samples(scm.clone(), 500)
    pd.testing.assert_frame_equal(original, cloned)


def test_random_nn_standardises_inputs_and_calibrates_output_percentiles():
    from dowhy.gcm.data_generator import _create_random_nn

    np.random.seed(0)
    nn = _create_random_nn(3, DataGeneratorConfig())
    X = np.random.randn(4000, 3) * np.array([0.01, 1.0, 100.0]) + np.array([5.0, 0.0, -300.0])
    nn.fit(X, np.zeros((4000, 1)))
    out = nn.predict(X)
    assert np.percentile(out, 1) == pytest.approx(-1.0, abs=1e-6)
    assert np.percentile(out, 99) == pytest.approx(1.0, abs=1e-6)
    hidden = np.tanh(((X - nn._in_mean) / nn._in_std) @ nn._weights[0] + nn._biases[0])
    assert np.mean(np.abs(hidden) > 0.99) < 0.25  # no step-like saturation despite wildly different input scales


def test_random_nn_handles_constant_and_discrete_inputs():
    from dowhy.gcm.data_generator import _create_random_nn

    np.random.seed(0)
    nn = _create_random_nn(2, DataGeneratorConfig())
    X = np.column_stack([np.full(1000, 3.0), np.random.randint(0, 3, 1000).astype(float)])
    nn.fit(X, np.zeros((1000, 1)))
    out = nn.predict(X)
    assert np.isfinite(out).all()
    assert len(np.unique(np.round(out, 6))) == 3  # one output level per discrete input level


def test_linear_coefficients_are_bounded_away_from_zero():
    from dowhy.gcm.data_generator import _create_random_linear

    np.random.seed(0)
    cfg = DataGeneratorConfig(linear_coefficient_range=(0.25, 1.0))
    for _ in range(200):
        w = np.abs(_create_random_linear(4, cfg)._weights[0].ravel())
        assert np.all(w >= 0.25) and np.all(w <= 1.0)


def test_all_linear_dense_scm_stays_bounded():
    # Regression: uncalibrated linear outputs grew to |x| ~ 80 along dense linear chains.
    cfg = DataGeneratorConfig(
        prob_linear_mechanism=1.0,
        prob_non_additive_noise=0.0,
        edge_density=1.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(3, 15, cfg), 2000)
    assert samples.abs().max().max() < 8


def test_discretised_mechanism_is_a_fixed_function():
    # Regression: bin edges were recomputed from each batch's min/max, so single rows and interventions were mis-binned.
    cfg = DataGeneratorConfig(
        prob_discretised=1.0,
        discrete_num_bins_range=(4, 4),
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_linear_mechanism=1.0,
        prob_non_additive_noise=0.0,
        noise_std_range=(1e-6, 1e-6),
    )
    np.random.seed(0)
    scm = generate_random_scm(1, 1, cfg)
    mechanism = scm.causal_mechanism("X1")
    parents = np.linspace(-3, 3, 601).reshape(-1, 1)
    noise = mechanism.draw_noise_samples(601)
    full = mechanism.evaluate(parents, noise).squeeze()
    single_rows = np.array([mechanism.evaluate(parents[[i]], noise[[i]]).squeeze() for i in range(0, 601, 50)])
    np.testing.assert_array_equal(single_rows, full[::50])
    assert set(np.unique(full)) <= {0.0, 1.0, 2.0, 3.0}
    assert len(np.unique(full)) >= 2


def test_discretised_roots_have_equal_frequency_bins():
    cfg = DataGeneratorConfig(
        prob_discretised=1.0, discrete_num_bins_range=(4, 4), prob_clipped_positive=0.0, prob_clipped_negative=0.0
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(4, 0, cfg), 4000)
    for col in samples.columns:
        frequencies = samples[col].value_counts(normalize=True)
        assert len(frequencies) == 4
        assert frequencies.min() > 0.2 and frequencies.max() < 0.3


def test_clipped_positive_nodes_have_controlled_zero_fraction():
    # Regression: clipping at literal 0 on zero-centred outputs gave ~50% zeros and near-constant nodes.
    cfg = DataGeneratorConfig(
        prob_clipped_positive=1.0, prob_clipped_negative=0.0, prob_discretised=0.0, clip_zero_fraction_range=(0.2, 0.2)
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(3, 6, cfg), 5000)
    for col in samples.columns:
        assert samples[col].min() >= 0
        assert 0.1 <= (samples[col] == 0).mean() <= 0.35


def test_clipped_negative_nodes_have_controlled_zero_fraction():
    cfg = DataGeneratorConfig(
        prob_clipped_positive=0.0, prob_clipped_negative=1.0, prob_discretised=0.0, clip_zero_fraction_range=(0.2, 0.2)
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(3, 6, cfg), 5000)
    for col in samples.columns:
        assert samples[col].max() <= 0
        assert 0.1 <= (samples[col] == 0).mean() <= 0.35


def test_transformed_mechanisms_are_functional_causal_models():
    from dowhy.gcm.causal_mechanisms import FunctionalCausalModel

    cfg = DataGeneratorConfig(prob_clipped_positive=0.5, prob_clipped_negative=0.0, prob_discretised=0.5)
    np.random.seed(0)
    scm = generate_random_scm(2, 4, cfg)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            mechanism = scm.causal_mechanism(node)
            assert isinstance(mechanism, FunctionalCausalModel)
            parents = np.random.randn(10, len(list(scm.graph.predecessors(node))))
            assert mechanism.evaluate(parents, mechanism.draw_noise_samples(10)).shape == (10, 1)
    np.random.seed(1)
    a = draw_samples(scm, 200)
    np.random.seed(1)
    b = draw_samples(scm.clone(), 200)
    pd.testing.assert_frame_equal(a, b)  # clones keep the frozen thresholds and bin edges


@pytest.mark.parametrize("family", ["gaussian", "uniform", "laplace", "student_t"])
def test_unimodal_noise_families_match_requested_std(family):
    from dowhy.gcm.data_generator import _create_noise_model

    np.random.seed(0)
    cfg = DataGeneratorConfig(noise_std_range=(0.3, 0.3), prob_unimodal_noise=1.0, unimodal_noise_weights={family: 1.0})
    x = np.concatenate([_create_noise_model(cfg).draw_samples(50000).squeeze() for _ in range(5)])
    assert abs(x.mean()) < 0.01
    assert x.std() == pytest.approx(0.3, rel=0.15)


def test_cauchy_noise_is_clipped_with_nominal_scale():
    from dowhy.gcm.data_generator import _create_noise_model

    np.random.seed(0)
    cfg = DataGeneratorConfig(
        noise_std_range=(0.1, 0.1), prob_unimodal_noise=1.0, unimodal_noise_weights={"cauchy": 1.0}
    )
    x = _create_noise_model(cfg).draw_samples(20000).squeeze()
    assert np.abs(x).max() <= 5.0 + 1e-12  # clipped at 50 * std
    assert np.median(np.abs(x)) == pytest.approx(0.06745, rel=0.1)  # IQR matched to a Gaussian with std 0.1


def test_unknown_noise_family_raises():
    with pytest.raises(ValueError):
        DataGeneratorConfig(unimodal_noise_weights={"gaussian": 0.5, "bogus": 0.5})
    with pytest.raises(ValueError):
        DataGeneratorConfig(unimodal_noise_weights={})


@pytest.mark.parametrize(
    "family", ["gaussian", "uniform", "laplace", "lognormal", "exponential", "beta", "student_t", "chi2"]
)
def test_root_families_are_standardised(family):
    from dowhy.gcm.data_generator import _create_root_model

    cfg = DataGeneratorConfig(
        prob_unimodal_root=1.0,
        root_distribution_weights={family: 1.0},
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
    )
    np.random.seed(0)
    x = _create_root_model(cfg).draw_samples(100000).squeeze()
    assert abs(x.mean()) < 0.03
    assert x.std() == pytest.approx(1.0, abs=0.1)


def test_skewed_root_with_negative_clip_is_not_constant():
    cfg = DataGeneratorConfig(
        prob_unimodal_root=1.0,
        root_distribution_weights={"exponential": 1.0},
        prob_clipped_positive=0.0,
        prob_clipped_negative=1.0,
        prob_discretised=0.0,
        clip_zero_fraction_range=(0.3, 0.3),
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(3, 0, cfg), 5000)
    for col in samples.columns:
        assert samples[col].max() <= 0
        assert 0.2 <= (samples[col] == 0).mean() <= 0.4
        assert samples[col].std() > 0.1


def test_unknown_root_family_raises():
    with pytest.raises(ValueError):
        DataGeneratorConfig(root_distribution_weights={"gaussian": 1.0, "pareto": 1.0})


def test_heteroscedastic_mechanism_is_invertible_and_varies_noise_scale():
    from dowhy.gcm.causal_mechanisms import InvertibleFunctionalCausalModel
    from dowhy.gcm.data_generator import _HeteroscedasticANM

    cfg = DataGeneratorConfig(
        prob_heteroscedastic_noise=1.0,
        prob_log_space_mechanism=0.0,
        prob_non_additive_noise=0.0,
        prob_linear_mechanism=1.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
        noise_std_range=(0.3, 0.3),
    )
    np.random.seed(0)
    n_varying = 0
    for _ in range(10):
        mechanism = generate_random_scm(1, 1, cfg).causal_mechanism("X1")
        assert isinstance(mechanism, _HeteroscedasticANM)
        assert isinstance(mechanism, InvertibleFunctionalCausalModel)
        X = np.random.randn(20000, 1)
        N = mechanism.draw_noise_samples(20000)
        Y = mechanism.evaluate(X, N)
        np.testing.assert_allclose(mechanism.estimate_noise(Y, X), N, atol=1e-8)
        residual = (Y - shape_into_2d(mechanism.prediction_model.predict(X))).squeeze()
        bins = np.digitize(X.squeeze(), np.quantile(X, [0.2, 0.4, 0.6, 0.8]))
        stds = [residual[bins == b].std() for b in range(5)]
        n_varying += int(max(stds) / min(stds) > 1.3)
    assert n_varying >= 7  # a random NN scale function is flat only occasionally


def test_default_config_produces_heteroscedastic_mechanisms_and_supports_counterfactuals():
    from dowhy import gcm
    from dowhy.gcm.data_generator import _HeteroscedasticANM

    np.random.seed(0)
    scm = generate_random_scm(3, 30)
    mechanisms = [scm.causal_mechanism(n) for n in scm.graph.nodes if not is_root_node(scm.graph, n)]
    assert any(isinstance(m, _HeteroscedasticANM) for m in mechanisms)

    cfg = DataGeneratorConfig(
        prob_heteroscedastic_noise=1.0,
        prob_log_space_mechanism=0.0,
        prob_non_additive_noise=0.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
    )
    np.random.seed(1)
    scm = generate_random_scm(1, 2, cfg)
    observed = draw_samples(scm, 5)
    # counterfactual_samples requires the InvertibleStructuralCausalModel type; the mechanisms live on the graph.
    invertible_scm = gcm.InvertibleStructuralCausalModel(scm.graph)
    counterfactuals = gcm.counterfactual_samples(invertible_scm, {"X0": lambda x: x + 1}, observed_data=observed)
    assert counterfactuals.shape == observed.shape


def test_log_space_mechanism_is_positive_consistent_and_bounded():
    from dowhy.gcm.causal_mechanisms import PostNonlinearModel
    from dowhy.gcm.data_generator import _ClippedExponentialFunction

    cfg = DataGeneratorConfig(
        prob_log_space_mechanism=1.0,
        prob_heteroscedastic_noise=0.0,
        prob_non_additive_noise=0.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
        noise_std_range=(0.2, 0.2),
    )
    np.random.seed(0)
    scm = generate_random_scm(2, 4, cfg)
    samples = draw_samples(scm, 3000)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            mechanism = scm.causal_mechanism(node)
            assert isinstance(mechanism, PostNonlinearModel)
            assert isinstance(mechanism.invertible_function, _ClippedExponentialFunction)
            assert (samples[node] > 0).all()
            assert np.isfinite(samples[node]).all()
            assert samples[node].max() <= np.exp(5) + 1e-9
    # Refitting on its own data reproduces the noise level (regression for the fork's log1p-vs-exp mismatch).
    node = "X3"
    parents = get_ordered_predecessors(scm.graph, node)
    refit = scm.causal_mechanism(node).clone()
    refit.fit(samples[parents].to_numpy(), samples[node].to_numpy())
    residuals = refit.estimate_noise(samples[node].to_numpy(), samples[parents].to_numpy())
    assert residuals.std() == pytest.approx(0.2, rel=0.25)


def test_heteroscedastic_and_log_space_probabilities_must_sum_to_at_most_one():
    with pytest.raises(ValueError):
        DataGeneratorConfig(prob_heteroscedastic_noise=0.6, prob_log_space_mechanism=0.5)


def test_generated_mechanisms_estimate_noise_with_correct_shape():
    # Regression: the random NN returned 1-D predictions, so PostNonlinearModel.estimate_noise broadcast (n, 1) - (n,)
    # into an (n, n) matrix for every generated additive mechanism.
    cfg = DataGeneratorConfig(
        prob_non_additive_noise=0.0,
        prob_heteroscedastic_noise=0.0,
        prob_clipped_positive=0.0,
        prob_clipped_negative=0.0,
        prob_discretised=0.0,
    )
    np.random.seed(0)
    scm = generate_random_scm(2, 4, cfg)
    samples = draw_samples(scm, 300)
    for node in scm.graph.nodes:
        if not is_root_node(scm.graph, node):
            mechanism = scm.causal_mechanism(node)
            X = samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
            assert mechanism.prediction_model.predict(X).shape == (300, 1)
            assert mechanism.estimate_noise(samples[node].to_numpy(), X).shape == (300, 1)


def test_total_edge_density_mode_gives_position_independent_in_degree():
    np.random.seed(0)
    dags = [generate_random_dag(3, 20, 0.2, edge_density_mode="total") for _ in range(20)]
    in_degrees = np.array([[dag.in_degree(f"X{i}") for i in range(3, 23)] for dag in dags])
    # Every child with enough predecessors gets 1 + round(0.2 * (23 - 2)) = 5 parents (plus at most a few repair
    # edges from otherwise isolated roots), independent of its position in the topological order.
    assert np.all(in_degrees[:, 2:] >= 5) and np.all(in_degrees[:, 2:] <= 8)
    assert abs(in_degrees[:, 5].mean() - in_degrees[:, 19].mean()) < 0.5
    assert all(nx.is_directed_acyclic_graph(dag) and nx.is_weakly_connected(dag) for dag in dags)


def test_preceding_mode_is_default_and_unchanged():
    np.random.seed(0)
    a = generate_random_dag(5, 10, edge_density=1.0)
    assert a.number_of_edges() == sum(range(5, 15))
    with pytest.raises(ValueError):
        generate_random_dag(3, 3, edge_density_mode="bogus")
    with pytest.raises(ValueError):
        DataGeneratorConfig(edge_density_mode="bogus")


@pytest.mark.parametrize("seed", range(10))
def test_default_config_data_is_well_behaved(seed):
    np.random.seed(seed)
    scm = generate_random_scm(4, 12)
    samples = draw_samples(scm, 2000)
    values = samples.to_numpy()
    assert np.isfinite(values).all()
    assert np.abs(values).max() < 150  # log-space nodes are capped at exp(5)
    for col in samples.columns:
        x = samples[col]
        assert x.nunique() >= 2  # no collapsed node
        if x.nunique() > 20:  # continuous node: neither collapsed nor exploded
            assert 0.05 < x.std() < 10


def test_default_scm_can_be_refit_and_cloned():
    from dowhy import gcm

    np.random.seed(0)
    scm = generate_random_scm(3, 8)
    samples = draw_samples(scm, 1000)
    clone = scm.clone()
    gcm.fit(clone, samples)
    refit_samples = draw_samples(clone, 1000)
    assert refit_samples.shape == (1000, 11)
    assert np.isfinite(refit_samples.to_numpy()).all()


def test_clipped_and_discretised_node_uses_all_bins_from_zero():
    # Regression: with a positive clip the point mass at 0 put a bin edge exactly at 0, and np.digitize (right=False)
    # sent those values to bin 1, so bin 0 was never populated and labels ran 1..bins.
    cfg = DataGeneratorConfig(
        prob_clipped_positive=1.0,
        prob_clipped_negative=0.0,
        prob_discretised=1.0,
        discrete_num_bins_range=(4, 4),
        clip_zero_fraction_range=(0.4, 0.4),
    )
    np.random.seed(0)
    samples = draw_samples(generate_random_scm(3, 0, cfg), 4000)
    for col in samples.columns:
        assert samples[col].min() == 0
        assert samples[col].nunique() == 4
        assert (samples[col] == 0).mean() == pytest.approx(0.4, abs=0.05)


def test_generated_graph_uses_plain_string_node_keys():
    # Regression: parents drawn via np.random.choice were stored as np.str_, which leaked into predecessors() and
    # into the keys of downstream results such as arrow_strength.
    np.random.seed(0)
    dag = generate_random_dag(3, 10)
    for node in dag.nodes:
        assert all(type(p) is str for p in dag.predecessors(node))
        assert all(type(c) is str for c in dag.successors(node))
