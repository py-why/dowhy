"""Random causal model and data generator.

Generates arbitrary-size DAGs with configurable data-generating processes:

- **Causal mechanisms**: Mix of linear functions and random neural networks with random weights.
  The NNs use ``tanh`` activations with spectral-normalised weights scaled to drive activations
  into the nonlinear regime, producing complex non-monotonic relationships.
- **Noise integration**: Either additive (``Y = f(X) + N``, an ANM) or non-additive
  (``Y = nn([X, N])``, where noise is fed through a random NN together with parents).
- **Noise distributions**: Mix of simple unimodal distributions (Gaussian, uniform) and
  multi-modal Gaussian mixtures.
- **Output transforms**: Nodes can be randomly clipped to positive-only or negative-only values,
  and/or discretised into a small number of bins.
- **Value stabilisation**: NN outputs are normalised to a configurable range via a calibration
  pass, preventing value collapse or explosion as signals propagate through the graph.

All behaviour is controlled through :class:`DataGeneratorConfig`.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from dowhy.gcm.causal_mechanisms import (
    AdditiveNoiseModel,
    ConditionalStochasticModel,
    FunctionalCausalModel,
    StochasticModel,
)
from dowhy.gcm.causal_models import PARENTS_DURING_FIT, StructuralCausalModel
from dowhy.gcm.fitting_sampling import draw_samples, fit
from dowhy.gcm.ml import PredictionModel
from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm.stochastic_models import ScipyDistribution
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node, validate_acyclic


@dataclass
class DataGeneratorConfig:
    """Controls all aspects of random SCM generation.

    **Graph structure**

    - ``edge_density``: Controls how many parents each non-root node gets.
      0 → exactly 1 parent per child (tree-like, sparse).
      1 → every child connects to all preceding nodes (maximally dense DAG).
      Default 0.15 gives ~2 parents per child on average.

    **Causal mechanisms (non-root nodes)**

    - ``prob_linear_mechanism``: Probability a non-root node uses a linear mechanism.
      The remaining ``1 - prob_linear_mechanism`` fraction uses a random neural network.
      Increasing this makes the SCM easier to learn; decreasing it adds more nonlinearity.
    - ``prob_non_additive_noise``: Probability that a non-root node uses a non-additive noise
      model (``Y = nn([X, N])``) instead of an additive one (``Y = f(X) + N``).
      At 0 every non-root node is an ANM; at 1 noise is always entangled with parents.
      Non-additive models are harder to identify and make the SCM non-invertible.
    - ``nn_hidden_units_range``: ``(min, max)`` number of hidden units per NN layer.
      Wider layers increase the expressiveness of nonlinear relationships.
    - ``nn_hidden_layers_range``: ``(min, max)`` number of hidden layers.
      More layers compose more nonlinearities, producing wilder functional forms.
    - ``nn_weight_scale``: Magnitude of spectral-normalised hidden weights.
      Higher values push ``tanh`` deeper into saturation, increasing nonlinearity.
    - ``nn_output_value_range``: ``(lo, hi)`` target range for NN outputs after calibration.
      Keeps values bounded regardless of network depth or fan-in.

    **Noise distributions**

    - ``prob_unimodal_noise``: Probability that noise is a simple unimodal distribution
      (Gaussian or uniform). The rest uses a Gaussian mixture, producing multi-modal noise.
    - ``noise_std_range``: ``(min, max)`` standard deviation for unimodal noise.
      Lower values make causal relationships cleaner; higher values add more stochasticity.
    - ``noise_mixture_components_range``: ``(min, max)`` number of Gaussian components in
      mixture noise/root distributions.
    - ``noise_mixture_component_std``: Standard deviation of each individual Gaussian component
      in mixture distributions (roots and noise).

    **Root-node distributions**

    - ``prob_unimodal_root``: Probability a root node uses a unimodal distribution (Gaussian or
      uniform). The rest uses a Gaussian mixture, producing multi-modal marginals.

    **Output transforms (applied per-node)**

    - ``prob_clipped_positive``: Probability a node's output is clipped at 0 from below
      (all values ≥ 0).
    - ``prob_clipped_negative``: Probability a node's output is clipped at 0 from above
      (all values ≤ 0). Mutually exclusive with ``prob_clipped_positive`` per node.
    - ``prob_discretised``: Probability a node's continuous output is discretised into bins.
    - ``discrete_num_bins_range``: ``(min, max)`` number of discrete bins when discretisation
      is applied.

    **Internal**

    - ``num_calibration_samples``: Number of samples used during assembly to calibrate NN output
      normalisation. Larger values give more stable calibration at the cost of speed.
    """

    # 0=sparse (1 parent each), 1=dense (all predecessors as parents)
    edge_density: float = 0.2

    # Fraction of non-root nodes using Y = wX + N instead of a random NN
    prob_linear_mechanism: float = 0.2
    # Fraction of non-root nodes where noise is entangled: Y = nn([X, N])
    prob_non_additive_noise: float = 0.2
    # (min, max) hidden units per NN layer
    nn_hidden_units_range: tuple = (4, 64)
    # (min, max) number of hidden layers in random NNs
    nn_hidden_layers_range: tuple = (2, 4)
    # Spectral-norm scale for hidden weights; higher = more nonlinear
    nn_weight_scale: float = 5.0
    # Target (lo, hi) for NN output normalisation
    nn_output_value_range: tuple = (-1.0, 1.0)

    # Fraction of noise distributions that are unimodal (Gaussian/uniform)
    prob_unimodal_noise: float = 0.5
    # (min, max) std for unimodal noise; controls noise magnitude
    noise_std_range: tuple = (0.01, 0.15)
    # (min, max) number of Gaussian components in mixture distributions
    noise_mixture_components_range: tuple = (2, 6)
    # Std of each Gaussian component in mixture distributions
    noise_mixture_component_std: float = 0.85

    # Fraction of root nodes using unimodal (Gaussian/uniform) vs mixture
    prob_unimodal_root: float = 0.4

    # Probability node output is clipped to ≥ 0
    prob_clipped_positive: float = 0.15
    # Probability node output is clipped to ≤ 0
    prob_clipped_negative: float = 0.05
    # Probability node output is discretised into bins
    prob_discretised: float = 0.05
    # (min, max) number of bins when discretising
    discrete_num_bins_range: tuple = (2, 8)

    # Samples used to calibrate NN output normalisation during assembly
    num_calibration_samples: int = 1000


def generate_samples_from_random_scm(
    num_roots: int, num_children: int, num_samples: int, config: Optional[DataGeneratorConfig] = None
) -> pd.DataFrame:
    """Generate a random SCM and return samples drawn from it.

    Convenience function that creates a random SCM, fits it to calibration data, and draws
    *num_samples* fresh samples.

    :param num_roots: Number of root nodes.
    :param num_children: Number of non-root nodes.
    :param num_samples: Number of samples to draw.
    :param config: Optional configuration. Uses defaults if *None*.
    :return: A DataFrame with *num_samples* rows and one column per node.
    """
    scm = generate_random_scm(num_roots, num_children, config)
    return draw_samples(scm, num_samples)


def generate_random_scm(
    num_roots: int, num_children: int, config: Optional[DataGeneratorConfig] = None
) -> StructuralCausalModel:
    """Generate a random SCM with a random DAG and random causal mechanisms.

    :param num_roots: Number of root nodes.
    :param num_children: Number of non-root nodes.
    :param config: Optional configuration. Uses defaults if *None*.
    :return: A randomly generated SCM.
    """
    if config is None:
        config = DataGeneratorConfig()
    return assign_random_fcms(generate_random_dag(num_roots, num_children, config.edge_density), config)


def generate_random_dag(num_roots: int, num_children: int, edge_density: float = 0.2) -> nx.DiGraph:
    """Generate a random DAG.

    :param num_roots: Number of root nodes.
    :param num_children: Number of non-root nodes.
    :param edge_density: 0 → each child gets exactly 1 parent (sparsest).
        1 → each child connects to all preceding nodes (densest acyclic graph).
    """
    graph = nx.DiGraph()

    for i in range(num_roots):
        graph.add_node("X" + str(i))

    children = ["X" + str(i + num_roots) for i in range(num_children)]
    for child in children:
        all_nodes = list(graph.nodes)
        max_parents = len(all_nodes)
        n_parents = 1 + int(np.round(edge_density * (max_parents - 1)))
        n_parents = max(1, min(n_parents, max_parents))
        parents = np.random.choice(all_nodes, n_parents, replace=False)
        graph.add_node(child)
        for p in parents:
            graph.add_edge(p, child)

    # Ensure no root is isolated: connect any root with no outgoing edges to a random child.
    if num_children > 0:
        for i in range(num_roots):
            root = "X" + str(i)
            if graph.out_degree(root) == 0:
                graph.add_edge(root, np.random.choice(children))

    # Ensure the graph is weakly connected: merge any disconnected components.
    components = list(nx.weakly_connected_components(graph))
    if len(components) > 1:
        main = max(components, key=len)
        for comp in components:
            if comp is main:
                continue
            # Add edge from a node in main to a node in comp, respecting topological order
            # (lower index → higher index) to guarantee acyclicity.
            source = min(main, key=lambda n: int(n[1:]))
            target = max(comp, key=lambda n: int(n[1:]))
            if int(source[1:]) < int(target[1:]):
                graph.add_edge(source, target)
            else:
                graph.add_edge(target, source)

    return graph


def assign_random_fcms(graph: nx.DiGraph, config: Optional[DataGeneratorConfig] = None) -> StructuralCausalModel:
    """Assign random causal mechanisms to every node in *graph*.

    Root nodes get a random distribution; non-root nodes get either an additive noise model
    (``Y = f(X) + N``) or a non-additive noise model (``Y = nn([X, N])``).  A calibration pass
    ensures NN outputs stay bounded.
    """
    if config is None:
        config = DataGeneratorConfig()

    validate_acyclic(graph)
    scm = StructuralCausalModel(graph)
    data = pd.DataFrame()
    n_cal = config.num_calibration_samples

    for node in nx.topological_sort(scm.graph):
        if is_root_node(graph, node):
            model = _create_root_model(config)
            scm.set_causal_mechanism(node, model)
            data[node] = model.draw_samples(n_cal).squeeze()
        else:
            parents = get_ordered_predecessors(scm.graph, node)
            model = _create_non_root_model(len(parents), config)
            scm.set_causal_mechanism(node, model)

            parent_data = data[parents].to_numpy()

            base = model._base if isinstance(model, _TransformedConditionalModel) else model
            if isinstance(base, _NonAdditiveNoiseFCM):
                noise = base.draw_noise_samples(n_cal).squeeze()
                combined = np.column_stack([parent_data, shape_into_2d(noise)])
                base._nn.fit(X=combined, Y=np.zeros((n_cal, 1)))
            elif isinstance(base, AdditiveNoiseModel):
                if isinstance(base.prediction_model, _RandomNNPredictionModel):
                    base.prediction_model.fit(X=parent_data, Y=np.zeros((n_cal, 1)))

            data[node] = model.draw_samples(parent_data)

        scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(scm.graph, node)

    return scm


def _apply_transform(
    samples: np.ndarray, clip: Optional[str] = None, discrete_bins: Optional[int] = None
) -> np.ndarray:
    """Apply optional clipping and discretisation to *samples*."""
    if clip == "positive":
        samples = np.maximum(samples, 0)
    elif clip == "negative":
        samples = np.minimum(samples, 0)
    if discrete_bins is not None:
        edges = np.linspace(samples.min() - 1e-8, samples.max() + 1e-8, discrete_bins + 1)
        samples = (np.digitize(samples, edges) - 1).astype(float)
    return samples


def _pick_transform(config: DataGeneratorConfig) -> tuple:
    """Roll the dice for a per-node output transform. Returns ``(clip, discrete_bins)``."""
    clip = None
    r = np.random.random()
    if r < config.prob_clipped_positive:
        clip = "positive"
    elif r < config.prob_clipped_positive + config.prob_clipped_negative:
        clip = "negative"

    discrete_bins = None
    if np.random.random() < config.prob_discretised:
        lo, hi = config.discrete_num_bins_range
        discrete_bins = int(np.random.randint(lo, hi + 1))

    return clip, discrete_bins


class _TransformedStochasticModel(StochasticModel):
    """Wraps any :class:`StochasticModel` and applies an output transform to ``draw_samples``."""

    def __init__(self, base: StochasticModel, clip: Optional[str] = None, discrete_bins: Optional[int] = None) -> None:
        self._base = base
        self._clip = clip
        self._discrete_bins = discrete_bins

    def fit(self, X: np.ndarray) -> None:
        self._base.fit(X)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        samples = self._base.draw_samples(num_samples).squeeze()
        return shape_into_2d(_apply_transform(samples, self._clip, self._discrete_bins))

    def clone(self):
        return _TransformedStochasticModel(self._base.clone(), self._clip, self._discrete_bins)


class _TransformedConditionalModel(ConditionalStochasticModel):
    """Wraps any non-root mechanism and applies the transform to the full output after noise."""

    def __init__(
        self, base: ConditionalStochasticModel, clip: Optional[str] = None, discrete_bins: Optional[int] = None
    ) -> None:
        self._base = base
        self._clip = clip
        self._discrete_bins = discrete_bins

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._base.fit(X, Y)

    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        samples = self._base.draw_samples(parent_samples).squeeze()
        return shape_into_2d(_apply_transform(samples, self._clip, self._discrete_bins))

    def clone(self):
        return _TransformedConditionalModel(self._base.clone(), self._clip, self._discrete_bins)


def _wrap_stochastic(model: StochasticModel, clip: Optional[str], discrete_bins: Optional[int]) -> StochasticModel:
    if clip or discrete_bins is not None:
        return _TransformedStochasticModel(model, clip, discrete_bins)
    return model


def _wrap_conditional(
    model: ConditionalStochasticModel, clip: Optional[str], discrete_bins: Optional[int]
) -> ConditionalStochasticModel:
    if clip or discrete_bins is not None:
        return _TransformedConditionalModel(model, clip, discrete_bins)
    return model


class _GaussianMixtureDistribution(StochasticModel):
    """Univariate Gaussian mixture with uniform component weights."""

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        self._means = np.asarray(means, dtype=np.float64)
        self._stds = np.asarray(stds, dtype=np.float64)
        self._weights = np.ones(len(self._means)) / len(self._means)

    def fit(self, X: np.ndarray) -> None:
        pass

    def draw_samples(self, num_samples: int) -> np.ndarray:
        ids = np.random.choice(len(self._means), size=num_samples, p=self._weights)
        samples = np.array([np.random.normal(self._means[i], self._stds[i]) for i in ids])
        return shape_into_2d(samples)

    def clone(self):
        return _GaussianMixtureDistribution(self._means.copy(), self._stds.copy())


def _random_mixture(config: DataGeneratorConfig, center: float = 0.0) -> _GaussianMixtureDistribution:
    """Create a random Gaussian mixture centred around *center*."""
    lo, hi = config.noise_mixture_components_range
    k = np.random.randint(lo, hi + 1)
    means = np.random.uniform(-1, 1, size=k)
    means = means - means.mean() + center
    stds = np.full(k, config.noise_mixture_component_std)
    return _GaussianMixtureDistribution(means, stds)


class _RandomNNPredictionModel(PredictionModel):
    """Feed-forward network with random weights and biases.

    Hidden-layer weights are spectral-normalised then scaled by ``nn_weight_scale`` to push
    ``tanh`` activations into their nonlinear regime.  A calibration pass in ``fit`` maps the
    raw output to ``[output_lo, output_hi]``.
    """

    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray], output_lo: float, output_hi: float) -> None:
        self._weights = weights
        self._biases = biases
        self._lo = output_lo
        self._hi = output_hi
        self._shift: float = 0.0
        self._scale: float = 1.0

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        raw = self._forward(X)
        lo, hi = float(np.min(raw)), float(np.max(raw))
        span = hi - lo if hi - lo > 1e-8 else 1.0
        self._shift = lo
        self._scale = span

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._forward(X)
        normed = (raw - self._shift) / self._scale
        return normed * (self._hi - self._lo) + self._lo

    def clone(self):
        return _RandomNNPredictionModel(
            [w.copy() for w in self._weights], [b.copy() for b in self._biases], self._lo, self._hi
        )

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for w, b in zip(self._weights[:-1], self._biases[:-1]):
            h = np.tanh(h @ w + b)
        return (h @ self._weights[-1] + self._biases[-1]).squeeze()


def _create_random_nn(num_inputs: int, config: DataGeneratorConfig) -> _RandomNNPredictionModel:
    """Build a random NN with spectral-normalised hidden weights and random biases."""
    lo_h, hi_h = config.nn_hidden_units_range
    lo_l, hi_l = config.nn_hidden_layers_range
    n_hidden = np.random.randint(lo_l, hi_l + 1)

    dims = [num_inputs]
    for _ in range(n_hidden):
        dims.append(np.random.randint(lo_h, hi_h + 1))
    dims.append(1)

    weights, biases = [], []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        w = np.random.randn(d_in, d_out)
        if i < len(dims) - 2:
            sigma = np.linalg.svd(w, compute_uv=False)[0]
            w = w / max(sigma, 1e-8) * config.nn_weight_scale
        biases.append(np.random.uniform(-1, 1, size=(1, d_out)))
        weights.append(w)

    return _RandomNNPredictionModel(weights, biases, *config.nn_output_value_range)


class _NonAdditiveNoiseFCM(FunctionalCausalModel):
    """Functional causal model where noise is entangled with parents: ``Y = nn([X, N])``.

    Unlike an :class:`AdditiveNoiseModel`, the noise here is *not* separable from the causal
    effect of parents — it is concatenated with parent values and fed through a random NN.
    """

    def __init__(self, nn: _RandomNNPredictionModel, noise_model: StochasticModel) -> None:
        self._nn = nn
        self._noise_model = noise_model

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        noise = self._noise_model.draw_samples(X.shape[0]).squeeze()
        combined = np.column_stack([X, shape_into_2d(noise)])
        self._nn.fit(combined, Y)

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self._noise_model.draw_samples(num_samples)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        combined = np.column_stack([shape_into_2d(parent_samples), shape_into_2d(noise_samples)])
        return self._nn.predict(combined)

    def clone(self):
        return _NonAdditiveNoiseFCM(self._nn.clone(), self._noise_model.clone())


def _create_root_model(config: DataGeneratorConfig) -> StochasticModel:
    """Return a random root-node distribution with an optional output transform baked in."""
    clip, disc = _pick_transform(config)

    if np.random.random() < config.prob_unimodal_root:
        if np.random.random() < 0.5:
            base = ScipyDistribution(stats.norm, loc=0, scale=1)
        else:
            base = ScipyDistribution(stats.uniform, loc=-np.sqrt(3), scale=2 * np.sqrt(3))
    else:
        base = _random_mixture(config)

    return _wrap_stochastic(base, clip, disc)


def _create_noise_model(config: DataGeneratorConfig) -> StochasticModel:
    """Return a random noise distribution (never transformed — transforms apply to the output)."""
    lo, hi = config.noise_std_range
    if np.random.random() < config.prob_unimodal_noise:
        std = np.random.uniform(lo, hi)
        if np.random.random() < 0.5:
            return ScipyDistribution(stats.norm, loc=0, scale=std)
        else:
            return ScipyDistribution(stats.uniform, loc=-std, scale=2 * std)
    else:
        mixture = _random_mixture(config, center=0.0)
        scale = np.random.uniform(lo, hi)
        mixture._stds = mixture._stds * scale / config.noise_mixture_component_std
        return mixture


def _create_non_root_model(
    num_inputs: int, config: DataGeneratorConfig
) -> Union[AdditiveNoiseModel, _NonAdditiveNoiseFCM]:
    """Return a random causal mechanism for a non-root node.

    With probability ``prob_non_additive_noise``, returns a :class:`_NonAdditiveNoiseFCM`
    (``Y = nn([X, N])``). Otherwise returns an :class:`AdditiveNoiseModel` (``Y = f(X) + N``).
    """
    noise = _create_noise_model(config)
    clip, disc = _pick_transform(config)

    if np.random.random() < config.prob_non_additive_noise:
        nn = _create_random_nn(num_inputs + 1, config)  # +1 for the noise dimension
        mechanism = _NonAdditiveNoiseFCM(nn, noise)
    elif np.random.random() < config.prob_linear_mechanism:
        reg = LinearRegression()
        reg.coef_ = np.random.uniform(-1, 1, num_inputs)
        reg.intercept_ = 0.0
        mechanism = AdditiveNoiseModel(SklearnRegressionModel(reg), noise_model=noise)
    else:
        mechanism = AdditiveNoiseModel(_create_random_nn(num_inputs, config), noise_model=noise)

    return _wrap_conditional(mechanism, clip, disc)
