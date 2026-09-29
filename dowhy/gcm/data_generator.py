"""Random causal model and data generator.

Generates arbitrary-size DAGs with configurable data-generating processes:

- **Causal mechanisms**: Mix of linear functions and random neural networks with random weights. The NNs use ``tanh``
  activations with spectral-normalised weights scaled to drive activations into the nonlinear regime, producing complex
  non-monotonic relationships. Inputs are standardised and outputs calibrated to a target range, for linear and NN
  mechanisms alike.
- **Noise integration**: Additive (``Y = f(X) + N``), heteroscedastic (``Y = f(X) + g(X) * N``), log-space
  multiplicative (``Y = exp(f(log X) + N)``) or non-additive (``Y = nn([X, N])``).
- **Noise distributions**: Gaussian, Laplace, uniform, Student-t (optionally clipped Cauchy) with an exact standard
  deviation, and multi-modal Gaussian mixtures.
- **Root distributions**: Gaussian, uniform, Laplace, log-normal, exponential, beta, Student-t and chi-squared
  (standardised to zero mean and unit variance) and multi-modal Gaussian mixtures.
- **Output transforms**: Nodes can be censored at a per-node threshold (a controlled fraction of exact zeros) and/or
  discretised into equal-frequency bins. Thresholds are fixed once during assembly, so mechanisms stay deterministic
  functions of their inputs.
- **Value stabilisation**: Mechanism outputs are normalised to a configurable range via a calibration pass, preventing
  value collapse or explosion as signals propagate through the graph.

All behaviour is controlled through :class:`DataGeneratorConfig`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from dowhy.gcm.causal_mechanisms import (
    AdditiveNoiseModel,
    FunctionalCausalModel,
    InvertibleFunctionalCausalModel,
    PostNonlinearModel,
    StochasticModel,
)
from dowhy.gcm.causal_models import PARENTS_DURING_FIT, StructuralCausalModel
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.gcm.ml import PredictionModel
from dowhy.gcm.ml.regression import InvertibleExponentialFunction
from dowhy.gcm.stochastic_models import ScipyDistribution
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node, validate_acyclic


def _check_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _check_range(name: str, value: tuple, lowest: float) -> None:
    lo, hi = value
    if lo < lowest or lo > hi:
        raise ValueError(f"{name} must satisfy {lowest} <= lo <= hi, got {value}.")


_NOISE_FAMILIES = ("gaussian", "uniform", "laplace", "student_t", "cauchy")
_EDGE_DENSITY_MODES = ("preceding", "total")


def _standardised_scipy(distribution, **shape_parameters) -> ScipyDistribution:
    """Scipy distribution shifted and scaled to zero mean and unit variance (shape, skew and tails preserved)."""
    mean, var = distribution.stats(moments="mv", **shape_parameters)
    scale = 1.0 / float(np.sqrt(var))
    return ScipyDistribution(distribution, loc=-float(mean) * scale, scale=scale, **shape_parameters)


_ROOT_FAMILIES = {
    "gaussian": lambda: ScipyDistribution(stats.norm, loc=0, scale=1),
    "uniform": lambda: ScipyDistribution(stats.uniform, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
    "laplace": lambda: _standardised_scipy(stats.laplace),
    "lognormal": lambda: _standardised_scipy(stats.lognorm, s=np.random.uniform(0.3, 1.0)),
    "exponential": lambda: _standardised_scipy(stats.expon),
    "beta": lambda: _standardised_scipy(stats.beta, a=np.random.uniform(0.5, 5.0), b=np.random.uniform(0.5, 5.0)),
    "student_t": lambda: _standardised_scipy(stats.t, df=int(np.random.choice([5, 7, 10]))),
    "chi2": lambda: _standardised_scipy(stats.chi2, df=int(np.random.choice([1, 2, 3, 5]))),
}


def _check_weights(name: str, weights: Dict[str, float], allowed: tuple) -> None:
    if (
        not weights
        or any(key not in allowed for key in weights)
        or any(value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError(
            f"{name} must map a non-empty subset of {allowed} to non-negative weights with a positive sum, got {weights}."
        )


@dataclass
class DataGeneratorConfig:
    """Controls all aspects of random SCM generation.

    **Graph structure**

    - ``edge_density``: Controls how many parents each non-root node gets.
      0 → exactly 1 parent per child (tree-like, sparse).
      1 → every child connects to all preceding nodes (maximally dense DAG).
      Default 0.2 gives about 2 parents per child in a 10-node graph.
    - ``edge_density_mode``: ``"preceding"`` (default) gives a child ``1 + round(edge_density * (k - 1))`` parents,
      ``k`` being its number of preceding nodes, so the mean in-degree grows with graph size (~6 at 50 nodes for
      0.2). ``"total"`` gives every child ``1 + round(edge_density * (N - 2))`` parents, independent of its position.

    **Causal mechanisms (non-root nodes)**

    - ``prob_linear_mechanism``: Probability a non-root node uses a linear mechanism.
      Increasing this makes the SCM easier to learn; decreasing it adds more nonlinearity.
    - ``prob_non_additive_noise``: Probability a non-root node uses a non-additive noise model
      (``Y = nn([X, N])``). Non-additive models are harder to identify and make the SCM non-invertible.
      The two probabilities partition the non-root nodes; the remaining
      ``1 - prob_linear_mechanism - prob_non_additive_noise`` fraction uses a random neural network with
      additive noise (``Y = f(X) + N``). Their sum must not exceed 1.
    - ``prob_heteroscedastic_noise``: Probability an additive (linear or NN) mechanism gets heteroscedastic noise
      ``Y = f(X) + (|g(X)| + 0.1) * N`` with a second random NN ``g`` (a location-scale noise model).
    - ``prob_log_space_mechanism``: Probability an additive mechanism operates in log space,
      ``Y = exp(f(sign(X) log(1 + |X|)) + N)``: positive, right-skewed, multiplicative relations with
      log-normal-like noise (exponent clipped to ±5). Together with ``prob_heteroscedastic_noise`` at most 1.
    - ``linear_coefficient_range``: ``(min, max)`` absolute value of linear coefficients (sign random). A minimum
      above 0 keeps every edge detectable. Linear outputs are calibrated to ``nn_output_value_range`` like NN outputs.
    - ``nn_hidden_units_range``: ``(min, max)`` number of hidden units per NN layer.
      Wider layers increase the expressiveness of nonlinear relationships.
    - ``nn_hidden_layers_range``: ``(min, max)`` number of hidden layers.
      More layers compose more nonlinearities, producing wilder functional forms.
    - ``nn_weight_scale``: Magnitude of spectral-normalised hidden weights (inputs are standardised first).
      Higher values push ``tanh`` deeper into saturation, increasing nonlinearity.
    - ``nn_output_value_range``: ``(lo, hi)`` target range for mechanism outputs after calibration; the 1st and 99th
      percentiles of the calibration outputs are mapped onto it. Keeps values bounded regardless of depth or fan-in.

    **Noise distributions**

    - ``prob_unimodal_noise``: Probability that noise is a unimodal distribution. The rest uses a Gaussian mixture,
      producing multi-modal noise.
    - ``unimodal_noise_weights``: Relative weights of the unimodal families ``gaussian``, ``uniform``, ``laplace``,
      ``student_t`` (df 3, 5 or 7) and ``cauchy`` (nominal scale, clipped at 50 std; not in the default mix). All
      families except Cauchy have exactly the drawn standard deviation. Non-Gaussian noise helps identifiability.
    - ``noise_std_range``: ``(min, max)`` standard deviation for all noise distributions (unimodal and mixture).
      Lower values make causal relationships cleaner; higher values add more stochasticity.
    - ``noise_std_log_uniform``: Sample the std log-uniformly (equal mass per order of magnitude) instead of
      uniformly; useful when ``noise_std_range`` spans more than a decade.
    - ``noise_mixture_components_range``: ``(min, max)`` number of Gaussian components in
      mixture noise/root distributions.
    - ``noise_mixture_mean_range``: Component means are drawn from ``U(-r, r)``.
    - ``noise_mixture_component_std``: Standard deviation of each component. Mixtures are standardised to zero
      mean and unit variance, so only the ratio to ``noise_mixture_mean_range`` matters: 0.4 / 3.0 gives clearly
      separated modes; larger ratios blur into a single bump.

    **Root-node distributions**

    - ``prob_unimodal_root``: Probability a root node uses a unimodal family. The rest uses a Gaussian mixture,
      producing multi-modal marginals.
    - ``root_distribution_weights``: Relative weights of the families ``gaussian``, ``uniform``, ``laplace``,
      ``lognormal``, ``exponential``, ``beta``, ``student_t`` and ``chi2``. Every family is standardised to zero mean
      and unit variance, so skew and tails differ while the scale stays comparable.

    **Output transforms (applied per-node)**

    - ``prob_clipped_positive``: Probability a node's output is censored from below: values under a per-node
      threshold become exactly 0 and the rest is shifted so all values are ≥ 0.
    - ``prob_clipped_negative``: Mirror image (all values ≤ 0). Mutually exclusive with ``prob_clipped_positive``
      per node.
    - ``clip_zero_fraction_range``: ``(min, max)`` fraction of a clipped node's values that are exactly 0; the
      threshold is the corresponding quantile, fixed during assembly.
    - ``prob_discretised``: Probability a node's continuous output is discretised into approximately
      equal-frequency bins ``0..bins-1``, with bin edges fixed during assembly.
    - ``discrete_num_bins_range``: ``(min, max)`` number of discrete bins when discretisation is applied.

    **Internal**

    - ``num_calibration_samples``: Number of samples used during assembly to calibrate mechanisms (input
      standardisation, output range, clip thresholds, bin edges). Larger values give more stable calibration at the
      cost of speed.
    """

    # 0=sparse (1 parent each), 1=dense (all predecessors as parents)
    edge_density: float = 0.2
    # "preceding": a child's parent count is edge_density x its number of preceding nodes (grows with position and
    # graph size). "total": expected in-degree 1 + edge_density x (N - 2) for every child, independent of position.
    edge_density_mode: str = "preceding"

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
    # (min, max) absolute value of linear coefficients; the sign is random. min > 0 keeps every edge detectable.
    linear_coefficient_range: tuple = (0.25, 1.0)
    # Probability an additive (linear or NN) mechanism gets heteroscedastic noise Y = f(X) + (|g(X)| + 0.1) * N
    prob_heteroscedastic_noise: float = 0.2
    # Probability an additive mechanism operates in log space: Y = exp(f(sign(X) log(1 + |X|)) + N), i.e. positive,
    # right-skewed multiplicative relations with log-normal-like noise. Mutually exclusive with heteroscedastic noise.
    prob_log_space_mechanism: float = 0.1

    # Fraction of noise distributions that are unimodal (vs. Gaussian mixtures)
    prob_unimodal_noise: float = 0.5
    # Relative weights of the unimodal noise families: gaussian, uniform, laplace, student_t, cauchy (clipped)
    unimodal_noise_weights: Dict[str, float] = field(
        default_factory=lambda: {"gaussian": 0.35, "laplace": 0.25, "uniform": 0.20, "student_t": 0.20}
    )
    # (min, max) std for noise distributions (unimodal and mixture); controls noise magnitude
    noise_std_range: tuple = (0.01, 0.5)
    # Sample the noise std log-uniformly (equal mass per order of magnitude) instead of uniformly
    noise_std_log_uniform: bool = False
    # (min, max) number of Gaussian components in mixture distributions
    noise_mixture_components_range: tuple = (2, 6)
    # Std of each Gaussian component relative to the spread of the component means (see noise_mixture_mean_range)
    noise_mixture_component_std: float = 0.4
    # Component means are drawn from U(-noise_mixture_mean_range, noise_mixture_mean_range) before standardisation
    noise_mixture_mean_range: float = 3.0

    # Fraction of root nodes using a unimodal family vs. a Gaussian mixture
    prob_unimodal_root: float = 0.4
    # Relative weights of the unimodal root families (all standardised to zero mean and unit variance):
    # gaussian, uniform, laplace, lognormal, exponential, beta, student_t, chi2
    root_distribution_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "gaussian": 0.25,
            "uniform": 0.15,
            "laplace": 0.10,
            "lognormal": 0.15,
            "exponential": 0.10,
            "beta": 0.10,
            "student_t": 0.05,
            "chi2": 0.10,
        }
    )

    # Probability node output is clipped to ≥ 0
    prob_clipped_positive: float = 0.15
    # Probability node output is clipped to ≤ 0
    prob_clipped_negative: float = 0.05
    # Probability node output is discretised into bins
    prob_discretised: float = 0.05
    # (min, max) number of bins when discretising
    discrete_num_bins_range: tuple = (2, 8)
    # (min, max) fraction of a clipped node's values that are exactly 0 (the clip threshold is a per-node quantile)
    clip_zero_fraction_range: tuple = (0.1, 0.4)

    # Samples used to calibrate NN output normalisation during assembly
    num_calibration_samples: int = 1000

    def __post_init__(self) -> None:
        self._validate_probabilities()
        self._validate_ranges()
        self._validate_weights()

    def _validate_probabilities(self) -> None:
        for name in (
            "edge_density",
            "prob_linear_mechanism",
            "prob_non_additive_noise",
            "prob_unimodal_noise",
            "prob_unimodal_root",
            "prob_clipped_positive",
            "prob_clipped_negative",
            "prob_discretised",
            "prob_heteroscedastic_noise",
            "prob_log_space_mechanism",
        ):
            _check_probability(name, getattr(self, name))
        if self.prob_clipped_positive + self.prob_clipped_negative > 1:
            raise ValueError("prob_clipped_positive + prob_clipped_negative must not exceed 1.")
        if self.prob_non_additive_noise + self.prob_linear_mechanism > 1:
            raise ValueError("prob_non_additive_noise + prob_linear_mechanism must not exceed 1.")
        if self.prob_heteroscedastic_noise + self.prob_log_space_mechanism > 1:
            raise ValueError("prob_heteroscedastic_noise + prob_log_space_mechanism must not exceed 1.")

    def _validate_ranges(self) -> None:
        for name, lowest in (
            ("nn_hidden_units_range", 1),
            ("nn_hidden_layers_range", 1),
            ("noise_mixture_components_range", 1),
            ("discrete_num_bins_range", 2),
            ("linear_coefficient_range", 0),
            ("clip_zero_fraction_range", 0),
        ):
            _check_range(name, getattr(self, name), lowest)
        if self.clip_zero_fraction_range[1] >= 1:
            raise ValueError("clip_zero_fraction_range values must be below 1.")
        if self.noise_std_range[0] <= 0 or self.noise_std_range[0] > self.noise_std_range[1]:
            raise ValueError(f"noise_std_range must satisfy 0 < lo <= hi, got {self.noise_std_range}.")
        if not self.nn_output_value_range[0] < self.nn_output_value_range[1]:
            raise ValueError(f"nn_output_value_range must satisfy lo < hi, got {self.nn_output_value_range}.")
        for name in ("nn_weight_scale", "noise_mixture_component_std"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.num_calibration_samples < 2:
            raise ValueError("num_calibration_samples must be at least 2.")
        if self.noise_mixture_mean_range < 0:
            raise ValueError("noise_mixture_mean_range must be non-negative.")
        if self.edge_density_mode not in _EDGE_DENSITY_MODES:
            raise ValueError(f"edge_density_mode must be one of {_EDGE_DENSITY_MODES}, got '{self.edge_density_mode}'.")

    def _validate_weights(self) -> None:
        _check_weights("unimodal_noise_weights", self.unimodal_noise_weights, _NOISE_FAMILIES)
        _check_weights("root_distribution_weights", self.root_distribution_weights, tuple(_ROOT_FAMILIES))


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
    return assign_random_fcms(
        generate_random_dag(num_roots, num_children, config.edge_density, config.edge_density_mode), config
    )


def generate_random_dag(
    num_roots: int, num_children: int, edge_density: float = 0.2, edge_density_mode: str = "preceding"
) -> nx.DiGraph:
    """Generate a random DAG.

    :param num_roots: Number of root nodes. If ``num_children == 0`` the result is ``num_roots`` isolated nodes.
    :param num_children: Number of non-root nodes.
    :param edge_density: 0 → each child gets exactly 1 parent (sparsest). 1 → each child connects to all preceding
        nodes (densest acyclic graph). In between, see ``edge_density_mode``.
    :param edge_density_mode: ``"preceding"`` (default): a child gets ``1 + round(edge_density * (k - 1))`` parents
        where ``k`` is its number of preceding nodes, so the mean in-degree grows with graph size (~2 at 10 nodes, ~6
        at 50 for 0.2). ``"total"``: every child gets ``1 + round(edge_density * (N - 2))`` parents (capped by its
        predecessors), independent of its position.
    """
    if edge_density_mode not in _EDGE_DENSITY_MODES:
        raise ValueError(f"edge_density_mode must be one of {_EDGE_DENSITY_MODES}, got '{edge_density_mode}'.")

    graph = nx.DiGraph()

    for i in range(num_roots):
        graph.add_node("X" + str(i))

    children = ["X" + str(i + num_roots) for i in range(num_children)]
    for child in children:
        all_nodes = list(graph.nodes)
        graph.add_node(child)
        # If there are no preceding nodes yet (e.g. num_roots == 0 and this is the first child), the node simply becomes
        # a root; adding parents would otherwise sample from an empty set and raise.
        if not all_nodes:
            continue
        n_parents = _number_of_parents(edge_density, edge_density_mode, len(all_nodes), num_roots + num_children)
        parents = np.random.choice(all_nodes, n_parents, replace=False).tolist()
        for p in parents:
            graph.add_edge(p, child)

    if num_children > 0:
        # Ensure no root is isolated: connect any root with no outgoing edges to a random child.
        for i in range(num_roots):
            root = "X" + str(i)
            if graph.out_degree(root) == 0:
                graph.add_edge(root, str(np.random.choice(children)))
        _connect_components(graph)

    return graph


def _number_of_parents(edge_density: float, edge_density_mode: str, num_preceding: int, num_nodes: int) -> int:
    if edge_density_mode == "total":
        n_parents = 1 + int(np.round(edge_density * max(num_nodes - 2, 0)))
    else:
        n_parents = 1 + int(np.round(edge_density * (num_preceding - 1)))
    return max(1, min(n_parents, num_preceding))


def _connect_components(graph: nx.DiGraph) -> None:
    """Make the graph weakly connected by linking every component to the largest one, respecting topological order."""
    components = list(nx.weakly_connected_components(graph))
    if len(components) <= 1:
        return
    main = max(components, key=len)
    for comp in components:
        if comp is main:
            continue
        # Add edge from a node in main to a node in comp, respecting topological order (lower index → higher index) to
        # guarantee acyclicity.
        source = min(main, key=lambda n: int(n[1:]))
        target = max(comp, key=lambda n: int(n[1:]))
        if int(source[1:]) < int(target[1:]):
            graph.add_edge(source, target)
        else:
            graph.add_edge(target, source)


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
            if isinstance(model, _TransformedStochasticModel):
                model.calibrate(n_cal)
            scm.set_causal_mechanism(node, model)
            data[node] = model.draw_samples(n_cal).squeeze()
        else:
            parents = get_ordered_predecessors(scm.graph, node)
            model = _create_non_root_model(len(parents), config)
            parent_data = data[parents].to_numpy()
            _calibrate(model, parent_data)
            scm.set_causal_mechanism(node, model)
            data[node] = model.draw_samples(parent_data).squeeze()

        scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(scm.graph, node)

    return scm


class _OutputTransform:
    """Per-node output transform whose thresholds are frozen once by :meth:`calibrate`.

    ``clip='positive'`` censors values below the ``clip_quantile`` quantile to exactly 0 (``'negative'`` mirrors this
    from above); ``discrete_bins`` maps values to equal-frequency bin indices ``0..bins-1``. Freezing the thresholds
    keeps the mechanism a fixed function of its inputs across batches, interventions and single-row evaluations.
    """

    def __init__(self, clip: Optional[str], clip_quantile: Optional[float], discrete_bins: Optional[int]) -> None:
        self._clip = clip
        self._clip_quantile = clip_quantile
        self._discrete_bins = discrete_bins
        self._threshold: Optional[float] = None
        self._edges: Optional[np.ndarray] = None

    def calibrate(self, raw: np.ndarray) -> None:
        raw = np.asarray(raw, dtype=float).ravel()
        if self._clip == "positive":
            self._threshold = float(np.quantile(raw, self._clip_quantile))
        elif self._clip == "negative":
            self._threshold = float(np.quantile(raw, 1 - self._clip_quantile))
        if self._discrete_bins is not None:
            levels = np.linspace(0, 1, self._discrete_bins + 1)[1:-1]
            self._edges = np.unique(np.quantile(self._apply_clip(raw), levels))

    def apply(self, samples: np.ndarray) -> np.ndarray:
        if (self._clip is not None and self._threshold is None) or (
            self._discrete_bins is not None and self._edges is None
        ):
            raise RuntimeError("Output transform used before calibrate() was called.")
        samples = self._apply_clip(np.asarray(samples, dtype=float).ravel())
        if self._edges is not None:
            # A positive clip piles mass at exactly 0, which is also the lowest bin edge; right=True keeps it in bin 0.
            samples = np.digitize(samples, self._edges, right=self._clip == "positive").astype(float)
        return samples

    def _apply_clip(self, samples: np.ndarray) -> np.ndarray:
        if self._clip == "positive":
            return np.maximum(samples - self._threshold, 0)
        if self._clip == "negative":
            return np.minimum(samples - self._threshold, 0)
        return samples

    def clone(self):
        clone = _OutputTransform(self._clip, self._clip_quantile, self._discrete_bins)
        clone._threshold = self._threshold
        clone._edges = None if self._edges is None else self._edges.copy()
        return clone


def _pick_transform(config: DataGeneratorConfig) -> Optional[_OutputTransform]:
    """Roll the dice for a per-node output transform; ``None`` means the node output is untouched."""
    clip, clip_quantile = None, None
    r = np.random.random()
    if r < config.prob_clipped_positive:
        clip = "positive"
    elif r < config.prob_clipped_positive + config.prob_clipped_negative:
        clip = "negative"
    if clip is not None:
        clip_quantile = float(np.random.uniform(*config.clip_zero_fraction_range))

    discrete_bins = None
    if np.random.random() < config.prob_discretised:
        lo, hi = config.discrete_num_bins_range
        discrete_bins = int(np.random.randint(lo, hi + 1))

    if clip is None and discrete_bins is None:
        return None
    return _OutputTransform(clip, clip_quantile, discrete_bins)


class _TransformedStochasticModel(StochasticModel):
    """Wraps a root :class:`StochasticModel` and applies a frozen output transform to ``draw_samples``."""

    def __init__(self, base: StochasticModel, transform: _OutputTransform) -> None:
        self._base = base
        self._transform = transform

    def calibrate(self, num_samples: int) -> None:
        self._transform.calibrate(self._base.draw_samples(num_samples))

    def fit(self, X: np.ndarray) -> None:
        self._base.fit(X)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        return shape_into_2d(self._transform.apply(self._base.draw_samples(num_samples)))

    def clone(self):
        return _TransformedStochasticModel(self._base.clone(), self._transform.clone())


class _TransformedConditionalModel(FunctionalCausalModel):
    """Wraps a non-root FCM and applies a frozen output transform to the full output (after noise).

    Staying a :class:`FunctionalCausalModel` keeps ``evaluate``/``draw_noise_samples`` available to noise-based GCM
    algorithms; clipping and discretisation are not invertible, so counterfactual noise reconstruction is not offered.
    """

    def __init__(self, base: FunctionalCausalModel, transform: _OutputTransform) -> None:
        self._base = base
        self._transform = transform

    def calibrate(self, parent_data: np.ndarray) -> None:
        _calibrate(self._base, parent_data)
        self._transform.calibrate(self._base.draw_samples(parent_data))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._base.fit(X, Y)

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self._base.draw_noise_samples(num_samples)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        return shape_into_2d(self._transform.apply(self._base.evaluate(parent_samples, noise_samples)))

    def clone(self):
        return _TransformedConditionalModel(self._base.clone(), self._transform.clone())


class _ClippedStochasticModel(StochasticModel):
    """Wraps a :class:`StochasticModel` and clips ``draw_samples`` to ``[-max_abs, max_abs]`` (bounded heavy tails)."""

    def __init__(self, base: StochasticModel, max_abs: float) -> None:
        self._base = base
        self._max_abs = max_abs

    def fit(self, X: np.ndarray) -> None:
        self._base.fit(X)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        return shape_into_2d(np.clip(self._base.draw_samples(num_samples).squeeze(), -self._max_abs, self._max_abs))

    def clone(self):
        return _ClippedStochasticModel(self._base.clone(), self._max_abs)


def _wrap_stochastic(model: StochasticModel, transform: Optional[_OutputTransform]) -> StochasticModel:
    return model if transform is None else _TransformedStochasticModel(model, transform)


def _wrap_conditional(model: FunctionalCausalModel, transform: Optional[_OutputTransform]) -> FunctionalCausalModel:
    return model if transform is None else _TransformedConditionalModel(model, transform)


def _calibrate(model, parent_data: np.ndarray) -> None:
    """Run the calibration pass of a non-root mechanism on samples of its parents.

    Mechanisms defined in this module implement ``calibrate(parent_data)``. Plain dowhy additive / post-nonlinear
    models built here hold a :class:`_RandomNNPredictionModel` (possibly wrapped) whose ``fit`` ignores the target and
    only calibrates, so it is called with a dummy target.
    """
    if hasattr(model, "calibrate"):
        model.calibrate(parent_data)
    else:
        model.prediction_model.fit(parent_data, np.zeros((parent_data.shape[0], 1)))


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
        return shape_into_2d(np.random.normal(self._means[ids], self._stds[ids]))

    def clone(self):
        return _GaussianMixtureDistribution(self._means.copy(), self._stds.copy())


def _random_mixture(config: DataGeneratorConfig) -> _GaussianMixtureDistribution:
    """Create a random Gaussian mixture with zero mean and unit variance.

    Component means are spread over ``[-noise_mixture_mean_range, noise_mixture_mean_range]`` and each component has
    std ``noise_mixture_component_std``; the mixture is then standardised, so only the ratio of the two controls how
    multimodal the result is (3.0 / 0.4 gives clearly separated modes in ~90% of draws).
    """
    lo, hi = config.noise_mixture_components_range
    k = np.random.randint(lo, hi + 1)
    r = config.noise_mixture_mean_range
    means = np.random.uniform(-r, r, size=k)
    means = means - means.mean()
    stds = np.full(k, config.noise_mixture_component_std)
    total_std = np.sqrt(np.mean(means**2) + config.noise_mixture_component_std**2)
    return _GaussianMixtureDistribution(means / total_std, stds / total_std)


def _draw_noise_std(config: DataGeneratorConfig) -> float:
    lo, hi = config.noise_std_range
    if config.noise_std_log_uniform:
        return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
    return float(np.random.uniform(lo, hi))


class _RandomNNPredictionModel(PredictionModel):
    """Feed-forward network with random weights and biases.

    Hidden-layer weights are spectral-normalised then scaled by ``nn_weight_scale`` to push ``tanh`` activations into
    their nonlinear regime. ``fit`` is a calibration pass that ignores the target: the first ``num_standardised_inputs``
    input columns (all by default) are standardised with their mean and std, and the 1st/99th percentiles of the raw
    output are mapped to ``[output_lo, output_hi]``. Without hidden layers the model is a calibrated linear function.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        output_lo: float,
        output_hi: float,
        num_standardised_inputs: Optional[int] = None,
    ) -> None:
        self._weights = weights
        self._biases = biases
        self._lo = output_lo
        self._hi = output_hi
        self._num_standardised_inputs = num_standardised_inputs
        self._in_mean: Optional[np.ndarray] = None
        self._in_std: Optional[np.ndarray] = None
        self._shift: float = 0.0
        self._scale: float = 1.0

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        X = shape_into_2d(X)
        k = X.shape[1] if self._num_standardised_inputs is None else self._num_standardised_inputs
        std = X[:, :k].std(axis=0)
        self._in_mean = X[:, :k].mean(axis=0)
        self._in_std = np.where(std > 1e-8, std, 1.0)
        raw = self._forward(X)
        lo, hi = np.percentile(raw, [1, 99])
        self._shift = float(lo)
        self._scale = float(hi - lo) if hi - lo > 1e-8 else 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._forward(X)
        return shape_into_2d((raw - self._shift) / self._scale * (self._hi - self._lo) + self._lo)

    def clone(self):
        clone = _RandomNNPredictionModel(
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
            self._lo,
            self._hi,
            self._num_standardised_inputs,
        )
        clone._in_mean = None if self._in_mean is None else self._in_mean.copy()
        clone._in_std = None if self._in_std is None else self._in_std.copy()
        clone._shift, clone._scale = self._shift, self._scale
        return clone

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.array(shape_into_2d(X), dtype=float)
        if self._in_mean is not None:
            k = len(self._in_mean)
            h[:, :k] = (h[:, :k] - self._in_mean) / self._in_std
        for w, b in zip(self._weights[:-1], self._biases[:-1]):
            h = np.tanh(h @ w + b)
        return (h @ self._weights[-1] + self._biases[-1]).squeeze()


def _create_random_nn(
    num_inputs: int, config: DataGeneratorConfig, num_standardised_inputs: Optional[int] = None
) -> _RandomNNPredictionModel:
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

    return _RandomNNPredictionModel(weights, biases, *config.nn_output_value_range, num_standardised_inputs)


def _create_random_linear(num_inputs: int, config: DataGeneratorConfig) -> _RandomNNPredictionModel:
    """Calibrated linear mechanism: random signs, magnitudes in ``linear_coefficient_range``, no hidden layers."""
    lo, hi = config.linear_coefficient_range
    coefficients = np.random.choice([-1.0, 1.0], num_inputs) * np.random.uniform(lo, hi, num_inputs)
    return _RandomNNPredictionModel([coefficients.reshape(-1, 1)], [np.zeros((1, 1))], *config.nn_output_value_range)


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

    def calibrate(self, parent_data: np.ndarray) -> None:
        self.fit(parent_data, np.zeros((parent_data.shape[0], 1)))

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self._noise_model.draw_samples(num_samples)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        combined = np.column_stack([shape_into_2d(parent_samples), shape_into_2d(noise_samples)])
        return self._nn.predict(combined)

    def clone(self):
        return _NonAdditiveNoiseFCM(self._nn.clone(), self._noise_model.clone())


_HETERO_SCALE_FLOOR = 0.1


class _HeteroscedasticANM(InvertibleFunctionalCausalModel):
    """Location-scale model ``Y = f(X) + (|g(X)| + 0.1) * N`` with a random NN ``g`` modulating the noise scale.

    ``f`` is a calibrated random NN or linear function, ``g`` a second calibrated random NN (outputs in
    ``nn_output_value_range``), so the noise scale varies smoothly with the parents (fan-shaped, butterfly, one-sided
    patterns). Invertible: ``N = (Y - f(X)) / (|g(X)| + 0.1)``.
    """

    def __init__(self, prediction_model: PredictionModel, noise_model: StochasticModel, scale_model: PredictionModel):
        self._prediction_model = prediction_model
        self._noise_model = noise_model
        self._scale_model = scale_model

    def _noise_scale(self, parent_samples: np.ndarray) -> np.ndarray:
        return np.abs(self._scale_model.predict(parent_samples).squeeze()) + _HETERO_SCALE_FLOOR

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        X = shape_into_2d(parent_samples)
        prediction = self._prediction_model.predict(X).squeeze()
        return shape_into_2d(prediction + self._noise_scale(X) * np.asarray(noise_samples).squeeze())

    def estimate_noise(self, target_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        X = shape_into_2d(parent_samples)
        prediction = self._prediction_model.predict(X).squeeze()
        return shape_into_2d((np.asarray(target_samples).squeeze() - prediction) / self._noise_scale(X))

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self._noise_model.draw_samples(num_samples)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = shape_into_2d(X, Y)
        self._prediction_model.fit(X, Y)
        self._scale_model.fit(X, Y)
        self._noise_model.fit(self.estimate_noise(Y, X))

    def calibrate(self, parent_data: np.ndarray) -> None:
        zeros = np.zeros((parent_data.shape[0], 1))
        self._prediction_model.fit(parent_data, zeros)
        self._scale_model.fit(parent_data, zeros)

    def clone(self):
        return _HeteroscedasticANM(self._prediction_model.clone(), self._noise_model.clone(), self._scale_model.clone())

    @property
    def prediction_model(self) -> PredictionModel:
        return self._prediction_model

    @property
    def noise_model(self) -> StochasticModel:
        return self._noise_model


_LOG_SPACE_EXPONENT_CLIP = 5.0


def _signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class _SignedLog1pInputModel(PredictionModel):
    """Applies ``sign(x) * log(1 + |x|)`` to all inputs before delegating to an inner prediction model."""

    def __init__(self, inner: PredictionModel) -> None:
        self._inner = inner

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._inner.fit(_signed_log1p(shape_into_2d(X)), Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._inner.predict(_signed_log1p(shape_into_2d(X)))

    def clone(self):
        return _SignedLog1pInputModel(self._inner.clone())


class _ClippedExponentialFunction(InvertibleExponentialFunction):
    """``exp`` with the exponent clipped to ``[-5, 5]`` so multiplicative chains stay within a factor of ~150."""

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(X, -_LOG_SPACE_EXPONENT_CLIP, _LOG_SPACE_EXPONENT_CLIP))


def _create_root_model(config: DataGeneratorConfig) -> StochasticModel:
    """Return a random root-node distribution (unit variance) with an optional output transform baked in."""
    transform = _pick_transform(config)
    if np.random.random() < config.prob_unimodal_root:
        base = _ROOT_FAMILIES[_pick_weighted(config.root_distribution_weights)]()
    else:
        base = _random_mixture(config)
    return _wrap_stochastic(base, transform)


def _pick_weighted(weights: Dict[str, float]) -> str:
    names = list(weights)
    p = np.array([weights[name] for name in names], dtype=float)
    return str(np.random.choice(names, p=p / p.sum()))


def _create_unimodal_noise(family: str, std: float) -> StochasticModel:
    """Zero-mean noise with standard deviation *std* (a nominal scale for Cauchy, which has no variance)."""
    if family == "gaussian":
        return ScipyDistribution(stats.norm, loc=0, scale=std)
    if family == "uniform":
        half_width = std * np.sqrt(3)  # a uniform on [-h, h] has std h / sqrt(3)
        return ScipyDistribution(stats.uniform, loc=-half_width, scale=2 * half_width)
    if family == "laplace":
        return ScipyDistribution(stats.laplace, loc=0, scale=std / np.sqrt(2))
    if family == "student_t":
        df = int(np.random.choice([3, 5, 7]))
        return ScipyDistribution(stats.t, df=df, loc=0, scale=std * np.sqrt((df - 2) / df))
    if family == "cauchy":
        # 0.6745 = norm.ppf(0.75): the interquartile range matches a Gaussian with that std. Clipped at 50 std.
        return _ClippedStochasticModel(ScipyDistribution(stats.cauchy, loc=0, scale=0.6745 * std), max_abs=50 * std)
    raise ValueError(f"Unknown noise family '{family}'.")


def _create_noise_model(config: DataGeneratorConfig) -> StochasticModel:
    """Return a random zero-mean noise distribution with the drawn standard deviation."""
    std = _draw_noise_std(config)
    if np.random.random() < config.prob_unimodal_noise:
        return _create_unimodal_noise(_pick_weighted(config.unimodal_noise_weights), std)
    mixture = _random_mixture(config)
    return _GaussianMixtureDistribution(mixture._means * std, mixture._stds * std)


def _build_additive_mechanism(
    prediction_model: PredictionModel, noise: StochasticModel, num_inputs: int, config: DataGeneratorConfig
) -> FunctionalCausalModel:
    """``Y = f(X) + N``, or its heteroscedastic / log-space variant, chosen by one partitioned draw."""
    r = np.random.random()
    if r < config.prob_heteroscedastic_noise:
        return _HeteroscedasticANM(prediction_model, noise, _create_random_nn(num_inputs, config))
    if r < config.prob_heteroscedastic_noise + config.prob_log_space_mechanism:
        return PostNonlinearModel(_SignedLog1pInputModel(prediction_model), noise, _ClippedExponentialFunction())
    return AdditiveNoiseModel(prediction_model, noise_model=noise)


def _create_non_root_model(num_inputs: int, config: DataGeneratorConfig) -> FunctionalCausalModel:
    """Return a random causal mechanism for a non-root node.

    One draw partitions the nodes into non-additive (``Y = nn([X, N])``), linear-additive and NN-additive mechanisms
    according to ``prob_non_additive_noise`` and ``prob_linear_mechanism``; additive mechanisms may additionally get
    heteroscedastic noise (``prob_heteroscedastic_noise``) or operate in log space (``prob_log_space_mechanism``).
    """
    noise = _create_noise_model(config)
    transform = _pick_transform(config)

    r = np.random.random()
    if r < config.prob_non_additive_noise:
        nn = _create_random_nn(num_inputs + 1, config, num_standardised_inputs=num_inputs)  # noise column stays raw
        mechanism = _NonAdditiveNoiseFCM(nn, noise)
    elif r < config.prob_non_additive_noise + config.prob_linear_mechanism:
        mechanism = _build_additive_mechanism(_create_random_linear(num_inputs, config), noise, num_inputs, config)
    else:
        mechanism = _build_additive_mechanism(_create_random_nn(num_inputs, config), noise, num_inputs, config)

    return _wrap_conditional(mechanism, transform)
