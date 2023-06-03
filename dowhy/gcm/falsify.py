"""This module provides functionality to falsify a user-given DAG given observed data.

Functions in this module should be considered experimental, meaning there might be breaking API changes in the future.
"""
import warnings
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import dowhy.gcm.config as config
from dowhy.gcm.graph import DirectedGraph, get_ordered_predecessors
from dowhy.gcm.independence_test import kernel_based
from dowhy.gcm.util import plot
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.validation import _get_non_descendants

COLORS = list(mcolors.TABLEAU_COLORS.values())

# Constants for falsification of a given graph
FALSIFY_N_VIOLATIONS = "n_violations"
FALSIFY_N_TESTS = "n_tests"
FALSIFY_P_VALUE = "p_value"
FALSIFY_P_VALUES = "p_values"

FALSIFY_GIVEN_VIOLATIONS = FALSIFY_N_VIOLATIONS + " g_given"
FALSIFY_PERM_VIOLATIONS = FALSIFY_N_VIOLATIONS + " permutations"
FALSIFY_LOCAL_VIOLATION_INSIGHT = "local violations"

FALSIFY_METHODS = {
    "validate_lmc": "LMC",
    "validate_pd": "Faithfulness",
    "validate_parental_dsep": "tPa",
    "validate_causal_minimality": "Causal Minimality",
}

FALSIFY_VIOLATION_COLOR = "red"


@dataclass
class _PValuesMemory:
    """A class to store and access results of independence tests.
    This class is useful if, e.g. we validate LMC on graph X -> Y -> Z and also on the permuted nodes Z -> Y -> X. Since
    X ind Z | Y == Z ind X | Y, we only need to perform the conditional independence test once. We use frozensets of
    nodes since standard (immutable) sets are not hashable.
    """

    p_values: Dict[Tuple[FrozenSet, FrozenSet, FrozenSet], float] = field(default_factory=dict)

    def add_p_value(
        self,
        p_value: float,
        X: Union[Set, List, str],
        Y: Union[Set, List, str],
        Z: Optional[Union[Set, List, str]] = None,
    ):
        if not Z:
            Z = set()
        if not self.get_p_value(X, Y, Z):
            self.p_values[(_to_frozenset(X), _to_frozenset(Y), _to_frozenset(Z))] = p_value

    def get_p_value(
        self, X: Union[Set, List, str], Y: Union[Set, List, str], Z: Optional[Union[Set, List, str]] = None
    ):
        if not Z:
            Z = set()
        # Independence tests are symmetric
        if (_to_frozenset(X), _to_frozenset(Y), _to_frozenset(Z)) in self.p_values:
            return self.p_values[(_to_frozenset(X), _to_frozenset(Y), _to_frozenset(Z))]
        elif (_to_frozenset(Y), _to_frozenset(X), _to_frozenset(Z)) in self.p_values:
            return self.p_values[(_to_frozenset(Y), _to_frozenset(X), _to_frozenset(Z))]
        return

    def clear_placeholders(self, placeholder_val=-1):
        keys_to_rem = set()
        for k, v in self.p_values.items():
            if v == placeholder_val:
                keys_to_rem.add(k)
        for k in keys_to_rem:
            del self.p_values[k]

    def __contains__(self, item: Tuple[Union[Set, List, str], ...]) -> bool:
        X, Y = (_to_frozenset(i) for i in item[:2])
        if len(item) == 2 or item[2] is None:
            Z = frozenset()
        else:
            Z = _to_frozenset(item[2])
        return (X, Y, Z) in self.p_values or (Y, X, Z) in self.p_values


def validate_lmc(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    p_values_memory: Optional[_PValuesMemory] = None,
    independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
    conditional_independence_test: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = kernel_based,
    significance_level: float = 0.05,
    include_unconditional: bool = True,
    n_jobs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Union[int, Dict[str, float]]]:
    """
    Validate the local markov condition for a given directed graph. Return number of violations and p values for each
    node.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param p_values_memory: _PValuesMemory instance, where results of previously performed tests are stored.
    :param independence_test: Test to use for unconditional independencies (only used if include_unconditional=True)
    :param conditional_independence_test: Conditional independence test to use for checking local Markov condition.
    :param significance_level: Significance level for (conditional) independence tests.
    :param include_unconditional: Test also unconditional independencies of root nodes.
    :param n_jobs: Number of jobs to use for parallel execution of (conditional) independence tests.
    :return: Outcome of validation containing number of violations in the graph and p values/violation for each tuple
        (node, non_desc)
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    if p_values_memory is None:
        p_values_memory = _PValuesMemory()

    validation_summary = {FALSIFY_N_VIOLATIONS: 0, FALSIFY_N_TESTS: 0, FALSIFY_P_VALUES: dict()}

    # Find out which tests to do
    triples = _get_parental_triples(causal_graph, include_unconditional)
    to_test = []
    for node, non_desc, parents in triples:
        if not (node, non_desc, parents) in p_values_memory:
            to_test.append((node, non_desc, parents))
            p_values_memory.add_p_value(-1, node, non_desc, parents)  # Placeholder

    # Parallelize over tests
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(to_test))
    p_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_p_value)(
            data=data,
            X=node,
            Y=non_desc,
            Z=parents,
            independence_test=independence_test,
            conditional_independence_test=conditional_independence_test,
            seed=seed,
        )
        for (node, non_desc, parents), seed in zip(to_test, random_seeds)
    )

    # Gather results
    p_values_memory.clear_placeholders()  # Clear placeholders
    for i, (node, non_desc, parents) in enumerate(to_test):
        p_values_memory.add_p_value(p_values[i], node, non_desc, parents)

    # Summarize
    for node, non_desc, parents in triples:
        validation_summary[FALSIFY_N_TESTS] += 1
        lmc_p_value = p_values_memory.get_p_value(node, non_desc, parents)
        validation_summary[FALSIFY_P_VALUES][(node, non_desc)] = (lmc_p_value, lmc_p_value <= significance_level)
        if lmc_p_value <= significance_level:
            validation_summary[FALSIFY_N_VIOLATIONS] += 1
    return validation_summary


def validate_parental_dsep(
    causal_graph: DirectedGraph, causal_graph_reference: DirectedGraph, include_unconditional: bool = True, **kwargs
) -> Dict[str, int]:
    """
    Graphical criterion to evaluate which pairwise parental d-separations in `causal_graph` are violated, assuming
    `causal_graph_reference` is the ground truth graph. If none are violated, then both graphs lie in the same Markov
    equivalence class.
    Specifically we test:
        X _|_G' Y | Z and X _/|_G Y | Z for Y \in ND{X}^G', Z = PA{X}^G
    :param causal_graph: Causal graph for which to evaluate parental d-separations (G')
    :param causal_graph_reference: Causal graph where we test if d-separation holds (G)
    :param include_unconditional: Test also unconditional independencies of root nodes.
    :return: Validation summary with number of d-separations implied by `causal_graph` and number of times these are
        violated in the graph `causal_graph_reference`.
    """
    validation_summary = {FALSIFY_N_VIOLATIONS: 0, FALSIFY_N_TESTS: 0}

    triples = _get_parental_triples(causal_graph, include_unconditional)
    for node, non_desc, parents in triples:
        validation_summary[FALSIFY_N_TESTS] += 1
        if not nx.d_separated(causal_graph_reference, {node}, {non_desc}, set(parents)):
            validation_summary[FALSIFY_N_VIOLATIONS] += 1
    return validation_summary


def validate_pd(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    p_values_memory: Optional[_PValuesMemory] = None,
    n_pairs: int = -1,
    independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
    significance_level: float = 0.05,
    adjacent_only: bool = False,
    n_jobs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Union[int, Dict[tuple, float]]]:
    """
    Validate pairwise dependencies (pd) for a given causal graph and data. Test for each node if it is statistically
    dependent of all its ancestors.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param p_values_memory: _PValuesMemory object, where results of previously performed tests are stored.
    :param n_pairs: Evaluate dependencies for n_pairs <= all pairs in the DAG. If n_pairs=-1, evaluate dependencies for
        all (ancestor, node) pairs (default).
    :param independence_test: Independence test to use for checking pairwise dependencies.
    :param significance_level: Significance level for independence tests.
    :param adjacent_only: Only test adjacent node pairs.
    :param n_jobs: Number of jobs to use for parallel execution of (conditional) independence tests.
    :return: Summary dict: {n_violations: int, n_tests: int, p_values: {(ancestor, node): float, ...}}
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    if p_values_memory is None:
        p_values_memory = _PValuesMemory()

    pairs = [(ancestor, node) for node in causal_graph.nodes for ancestor in nx.ancestors(causal_graph, node)]
    if adjacent_only:
        pairs = [
            (ancestor, node) for (ancestor, node) in pairs if ancestor in get_ordered_predecessors(causal_graph, node)
        ]

    if n_pairs < 0:
        n_pairs = len(pairs)

    if n_pairs > len(pairs):
        raise ValueError(f"n_pairs ({n_pairs}) > number of pairs in the DAG ({len(pairs)})")

    validation_summary = {FALSIFY_N_VIOLATIONS: 0, FALSIFY_N_TESTS: n_pairs, FALSIFY_P_VALUES: dict()}
    pair_idxs = np.random.choice(len(pairs), size=n_pairs, replace=False)

    # Find out which tests to do
    to_test = []
    for pair_idx in pair_idxs:
        ancestor, node = pairs[pair_idx]
        if not (ancestor, node) in p_values_memory:
            to_test.append((ancestor, node))
            p_values_memory.add_p_value(-1, ancestor, node)  # Placeholder

    # Parallelize over tests
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(to_test))
    p_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_p_value)(
            data=data,
            X=ancestor,
            Y=node,
            Z=None,
            independence_test=independence_test,
            conditional_independence_test=None,
            seed=seed,
        )
        for (ancestor, node), seed in zip(to_test, random_seeds)
    )

    # Gather results
    p_values_memory.clear_placeholders()  # Clear placeholders
    for i, (ancestor, node) in enumerate(to_test):
        p_values_memory.add_p_value(p_values[i], ancestor, node)

    # Summarize
    for pair_idx in pair_idxs:
        ancestor, node = pairs[pair_idx]
        p_value = p_values_memory.get_p_value(ancestor, node)
        validation_summary[FALSIFY_P_VALUES][(node, ancestor)] = (p_value, p_value > significance_level)
        if p_value > significance_level:
            validation_summary[FALSIFY_N_VIOLATIONS] += 1

    return validation_summary


def validate_causal_minimality(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    p_values_memory: Optional[_PValuesMemory] = None,
    independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
    conditional_independence_test: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = kernel_based,
    significance_level: float = 0.05,
    n_jobs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Union[int, Dict[tuple, float]]]:
    """
    Function to test causal minimality of a DAG (see [1], Proposition 6.36).
    [1] J. Peters, D. Janzing, and B. Schölkopf, Elements of Causal Inference: Foundations and Learning Algorithms. Cambridge, MA, USA: MIT Press, 2017.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param p_values_memory: _PValuesMemory object, where results of previously performed tests are stored.
    :param independence_test: Independence test to use.
    :param conditional_independence_test: Conditional independence test to use.
    :param significance_level: Significance level for independence tests.
    :param n_jobs: Number of jobs to use for parallel execution of (conditional) independence tests.
    :return: Validation summary as dict.
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    if p_values_memory is None:
        p_values_memory = _PValuesMemory()

    validation_summary = {FALSIFY_N_VIOLATIONS: 0, FALSIFY_N_TESTS: 0, FALSIFY_P_VALUES: dict()}

    # Find out which tests to do
    triples = []
    to_test = []
    for node in causal_graph.nodes:
        parents = set(causal_graph.predecessors(node))
        if parents:
            for p in parents:
                other_parents = parents.difference({p})
                triples.append((node, p, other_parents))
                if not (node, p, other_parents) in p_values_memory:
                    to_test.append((node, p, other_parents))
                    p_values_memory.add_p_value(-1, node, p, other_parents)  # Placeholder

    # Parallelize over tests
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(to_test))
    p_values = Parallel(n_jobs=n_jobs)(
        delayed(_compute_p_value)(
            data=data,
            X=node,
            Y=p,
            Z=list(other_parents),
            independence_test=independence_test,
            conditional_independence_test=conditional_independence_test,
            seed=seed,
        )
        for (node, p, other_parents), seed in zip(to_test, random_seeds)
    )

    # Gather results
    p_values_memory.clear_placeholders()  # Clear placeholders
    for i, (node, p, other_parents) in enumerate(to_test):
        p_values_memory.add_p_value(p_values[i], node, p, other_parents)

    # Summarize
    for node, p, other_parents in triples:
        validation_summary[FALSIFY_N_TESTS] += 1
        p_value = p_values_memory.get_p_value(node, p, other_parents)
        validation_summary[FALSIFY_P_VALUES][(node, p, tuple(other_parents))] = (p_value, p_value > significance_level)
        if p_value > significance_level:
            validation_summary[FALSIFY_N_VIOLATIONS] += 1

    return validation_summary


def validate_graph(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    methods: Union[Callable, Tuple[Callable, ...], List[Callable]] = (validate_lmc, validate_pd),
    independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
    conditional_independence_test: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = kernel_based,
    significance_level: float = 0.05,
    p_values_memory: Optional[_PValuesMemory] = None,
    n_jobs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Generate baseline for node permutations.
    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param methods: Validation methods to perform. Supported are: validate_lmc, validate_ed.
    :param independence_test: (Unconditional) independence test to use.
    :param conditional_independence_test: Conditional independence test to use.
    :param significance_level: Significance level for (conditional) independence tests.
    :param p_values_memory: Optional _PValuesMemory object, where results of previously performed tests are stored.
    :param n_jobs: Number of jobs to use for parallel execution of (conditional) independence tests.
    :return: Validation summary as dict.
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    if p_values_memory is None:
        p_values_memory = _PValuesMemory()

    validation_summary = dict()

    if not isinstance(methods, (tuple, list)):
        methods = (methods,)

    for m in methods:
        # Call individual validation methods. Unused arguments are absorbed in their respective **kwargs.
        m_summary = m(
            causal_graph=causal_graph,
            data=data,
            p_values_memory=p_values_memory,
            independence_test=independence_test,
            conditional_independence_test=conditional_independence_test,
            significance_level=significance_level,
            n_jobs=n_jobs,
            **kwargs,
        )

        validation_summary[m.__name__] = m_summary

    return validation_summary


@dataclass
class EvaluationResult:
    """
    Dataset class containing the evaluation result of falsifying a graph using a node-permutation test.

    ...

    Attributes
    ----------
    methods : tuple
        Tuple containing the methods used for the node permutation test
    summary : dict
        Dictionary containing the summary of the evaluation.
    significance_level : float
        Significance level based on which we falsify the given DAG
    falsifiable : bool
        Whether the given DAG is falsifiable.
    falsified : bool
        Whether the given DAG is falsified.

    """

    methods: tuple
    summary: dict
    significance_level: float
    suggestions: Optional[dict] = None

    def update_significance_level(self, significance_level: float):
        """
        Update the significance level to decide if we falsify a given DAG.
        """
        self.significance_level = significance_level
        self.__post_init__()

    def __post_init__(self):
        self.can_evaluate = self._can_evaluate()
        if not self.can_evaluate:
            self.falsified = None
            self.falsifiable = None
        elif (
            self.summary["validate_lmc"][FALSIFY_P_VALUE]
            > self.significance_level
            > self.summary["validate_parental_dsep"][FALSIFY_P_VALUE]
        ):
            self.falsified = True
            self.falsifiable = True
        elif self.significance_level < self.summary["validate_parental_dsep"][FALSIFY_P_VALUE]:
            self.falsified = False
            self.falsifiable = False
        else:
            self.falsified = False
            self.falsifiable = True

    def __repr__(self):
        # DAG Evaluation
        if self.can_evaluate:
            decision = " " if self.falsified else " do not "
            informative = " " if self.falsifiable else " not "
            frac_MEC = (
                f"{len(self.summary['MEC'])} / {len(self.summary['validate_parental_dsep'][FALSIFY_PERM_VIOLATIONS])}"
            )
            frac_VLMC = f"{self.summary['validate_lmc'][FALSIFY_GIVEN_VIOLATIONS]}/{self.summary['validate_lmc'][FALSIFY_N_TESTS]}"
            p_LMC = self.summary["validate_lmc"][FALSIFY_P_VALUE]
            p_dSep = self.summary["validate_parental_dsep"][FALSIFY_P_VALUE]
            validation_repr = [
                f"The given DAG is{informative}informative because {frac_MEC} of the permutations lie in the Markov",
                f"equivalence class of the given DAG (p-value: {p_dSep:.2f}).",
                f"The given DAG violates {frac_VLMC} LMCs and is better than {(1 - p_LMC) * 100:.1f}% of the permuted DAGs (p-value: {p_LMC:.2f}).",
                f"Based on the provided significance level ({self.significance_level}) and because the DAG is{informative}informative,",
                f"we{decision}reject the DAG.",
            ]
        else:
            validation_repr = ["Cannot be evaluated!"]

        # Suggestions
        suggestion_repr = {}
        for m in self.suggestions:
            suggestion_repr[FALSIFY_METHODS[m]] = [
                f"Remove edge {node[1]} --> {node[0]}"
                for (node, r) in self.suggestions[m][FALSIFY_P_VALUES].items()
                if r[1]
            ]
        return _generate_table(validation_repr, suggestion_repr)

    def _can_evaluate(self):
        can_evaluate = True
        for m in (validate_lmc, validate_parental_dsep):
            if m not in self.methods:
                warnings.warn(f"Method {m.__name__} not in methods and thus graph cannot be evaluated!")
                can_evaluate = False
        return can_evaluate


def falsify_graph(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    methods: Union[Callable, Tuple[Callable, ...]] = (validate_lmc, validate_parental_dsep),
    suggestion_methods: Union[Callable, Tuple[Callable, ...]] = (validate_causal_minimality),
    suggestions: bool = False,
    independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
    conditional_independence_test: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = kernel_based,
    significance_level: float = 0.05,
    significance_ci: float = 0.05,
    n_permutations: Optional[int] = None,
    show_progress_bar: Optional[bool] = None,
    n_jobs: Optional[int] = None,
    plot_histogram: bool = False,
    plot_kwargs: Optional[Dict] = None,
) -> EvaluationResult:
    """
    Falsify a given DAG using observational data.

    This method returns the result of a permutation-test to falsify a user-given DAG using observational data. To this
    end we construct the test statistics by testing the violations of local Markov conditions (LMC) implied by
    the graph using conditional independence (CI) tests. The null is the number of LMC violations of a random
    node-permutation of the given graph. Our test can be interpreted as whether the given graph is significantly better
    than random in terms of the CIs it entails.
    To determine whether a given graph is falsifiable by our metric, we implement a second test, which reports whether
    given graph is "characteristic" enough in terms of the CIs it entails. For this, we compute how many of the random
    node permutations lie in the same Markov equivalence class (MEC) as the given graph and conclude that the given
    graph is falsifiable only if the fraction of permuted DAGs in the same MEC as the given graph is "reasonably" small.

    The returned EvaluationResult object has two attributes: `falsified` and `falsifiable`:
        `falsifiable`: The given graph lies in a different MEC than >= 1-`significance_level` of the permuted DAGs
        `falsified`: The given graph is falsifiable and violates fewer LMCs than >= 1-`significance_level` of the
                        permuted DAGs

    By default, we only run 1 / `significance_level` permutations as those are enough to falsify a graph with type I
    error probability `significance_level` at some given `significance_level`. If you are interested in a more exact
    estimate of the p-value of whish to plot a histogram to see how the given DAG compares to random node permutations,
    you should set `n_permutations` to some larger value (e.g. 100 or 1000). If `n_permutations=-1` we test on all
    n_nodes! permutations.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param methods: Validation methods to perform.
    :param suggestion_methods: Methods to run on the given graph to provide additional suggestions.
    :param suggestions: Provide suggestions generated using the `suggestion_methods`.
    :param independence_test: Independence test to use for checking pairwise independencies.
    :param conditional_independence_test: Conditional independence test to use.
    :param significance_level: Significance level for the permutation test.
    :param significance_ci: Significance level for (conditional) independence tests.
    :param n_permutations: Number of permutations to perform. If -1 use all n_nodes! permutations.
    :param show_progress_bar: Whether to show progress bar over permutations.
    :param n_jobs: Number of jobs to use for parallel execution of (conditional) independence tests.
    :param plot_histogram: Plot histogram of results from permutation baseline.
    :param plot_kwargs: Additional plot arguments to be passed to plot_evaluation_results.
    :return: EvaluationResult
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs
    show_progress_bar = config.show_progress_bars if show_progress_bar is None else show_progress_bar

    if n_permutations is None:
        n_permutations = int(1 / significance_level) if not plot_histogram else -1

    if not plot_kwargs:
        plot_kwargs = {}
    if isinstance(methods, Callable):
        methods = (methods,)
    if not suggestions:
        suggestion_methods = tuple()
    elif isinstance(suggestion_methods, Callable):
        suggestion_methods = (suggestion_methods,)

    p_values_memory = _PValuesMemory()

    summary_given = validate_graph(
        causal_graph,
        data=data,
        p_values_memory=p_values_memory,
        causal_graph_reference=causal_graph,
        methods=methods + suggestion_methods,
        independence_test=independence_test,
        conditional_independence_test=conditional_independence_test,
        significance_level=significance_ci,
        n_jobs=n_jobs,
    )
    summary_perm = _permutation_based(
        causal_graph,
        data=data,
        p_values_memory=p_values_memory,
        exclude_original_order=False,
        n_permutations=n_permutations,
        methods=methods,
        independence_test=independence_test,
        conditional_independence_test=conditional_independence_test,
        significance_level=significance_ci,
        show_progress_bar=show_progress_bar,
        n_jobs=n_jobs,
    )

    summary = {m.__name__: dict() for m in methods}

    for m, m_summary in summary.items():
        m_summary[FALSIFY_PERM_VIOLATIONS] = [perm[FALSIFY_N_VIOLATIONS] for perm in summary_perm[m]]
        m_summary[FALSIFY_GIVEN_VIOLATIONS] = summary_given[m][FALSIFY_N_VIOLATIONS]
        m_summary[FALSIFY_N_TESTS] = summary_given[m][FALSIFY_N_TESTS]
        m_summary[FALSIFY_P_VALUE] = sum(
            [1 for perm in m_summary[FALSIFY_PERM_VIOLATIONS] if perm <= m_summary[FALSIFY_GIVEN_VIOLATIONS]]
        ) / len(m_summary[FALSIFY_PERM_VIOLATIONS])

        if m != "validate_parental_dsep":
            # Append list of violations (node, non_desc) to get local information
            m_summary[FALSIFY_LOCAL_VIOLATION_INSIGHT] = summary_given[m][FALSIFY_P_VALUES]

    if "validate_parental_dsep" in summary:
        summary["MEC"] = [
            summary_perm["permuted_graphs"][i]
            for i, v in enumerate(summary["validate_parental_dsep"][FALSIFY_PERM_VIOLATIONS])
            if v == 0
        ]

    result = EvaluationResult(
        methods=methods,
        summary=summary,
        significance_level=significance_level,
        suggestions={m.__name__: summary_given[m.__name__] for m in suggestion_methods},
    )
    if plot_histogram:
        plot_evaluation_results(result, **plot_kwargs)
    return result


def apply_suggestions(
    causal_graph: DirectedGraph,
    evaluation_result: EvaluationResult,
    edges_to_keep: Optional[List[Tuple[Any, Any]]] = None,
):
    if not hasattr(evaluation_result, "suggestions"):
        raise ValueError("EvaluationResult object has no attribute suggestions. Please run with suggestion=True!")

    causal_graph = causal_graph.copy()
    for m in evaluation_result.suggestions:
        for node, res in evaluation_result.suggestions[m][FALSIFY_P_VALUES].items():
            edge = (node[1], node[0])
            if (res[1] and edges_to_keep is not None and edge not in edges_to_keep) or (
                res[1] and edges_to_keep is None
            ):
                causal_graph.remove_edge(edge[0], edge[1])
    return causal_graph


def plot_evaluation_results(
    evaluation_result, figsize=(8, 3), bins=None, maxbins=10, title="", savepath="", display=True
):
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    p_values = ""
    data = []
    labels = []

    evaluation_summary = {k: v for k, v in evaluation_result.summary.items() if k != "MEC"}
    for i, (m, m_summary) in enumerate(evaluation_summary.items()):
        data.append(m_summary[FALSIFY_PERM_VIOLATIONS])
        labels.append(f"Violations of {FALSIFY_METHODS[m]} of permuted DAGs")
        p_values += f"p-value {FALSIFY_METHODS[m]} = {m_summary[FALSIFY_P_VALUE]:.2f}\n"
    if bins is None:
        max_data = max(max(v) for v in data) + 0.5
        bins = np.linspace(-0.5, max_data, min(maxbins, int(max_data + 1.5)))
    ax.hist(data, color=COLORS[: len(evaluation_summary)], bins=bins, alpha=0.5, label=labels, edgecolor="k")

    # Plot given violations
    for i, (m, m_summary) in enumerate(evaluation_summary.items()):
        ylim = ax.get_ylim()[1]
        ax.plot(
            [m_summary[FALSIFY_GIVEN_VIOLATIONS]] * 2,
            [0, ylim],
            "--",
            c=COLORS[i],
            label=f"Violations of {FALSIFY_METHODS[m]} of given DAG",
        )
        ax.set_ylim([0, ylim])

    ax.set_xlabel("Violations")
    ax.set_ylabel("# Permutations")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0, title=p_values)
    if title:
        plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()


def plot_local_insights(
    causal_graph: DirectedGraph,
    evaluation_result: Union[EvaluationResult, Dict],
    method: Optional[str] = "validate_lmc",
):
    """
    For some given graph and evaluation result plot local violations.
    :param causal_graph: DiGraph
    :param evaluation_result: EvaluationResult
    :param method: Method for which to plot violations
    """
    colors = {}
    if isinstance(evaluation_result, EvaluationResult) and method in evaluation_result.summary:
        local_insight_dict = evaluation_result.summary[method][FALSIFY_LOCAL_VIOLATION_INSIGHT]
    elif (
        isinstance(evaluation_result, EvaluationResult)
        and hasattr(evaluation_result, "suggestions")
        and method in evaluation_result.suggestions
    ):
        local_insight_dict = evaluation_result.suggestions[method][FALSIFY_P_VALUES]
    elif isinstance(evaluation_result, Dict):
        if method not in evaluation_result:
            raise ValueError(f"Validation method {method} does not exist in given evaluation_result!")
        if FALSIFY_P_VALUES not in evaluation_result[method]:
            raise ValueError(
                f"Validation method {method} has no key {FALSIFY_P_VALUES} where information on local violations "
                f"are stored!"
            )
        local_insight_dict = evaluation_result[method][FALSIFY_P_VALUES]
    else:
        raise ValueError(f"Cannot plot local violation insights from method {method} in evaluation_result!")

    for nodes, result in local_insight_dict.items():
        if result[1]:
            if method == "validate_lmc":
                # For LMC we highlight X for which X _|/|_ Y \in ND_X | Pa_X
                colors[nodes[0]] = FALSIFY_VIOLATION_COLOR
            elif method == "validate_pd":
                # For PD we highlight the edge (if Y\in Anc_X -> X are adjacent)
                colors[(nodes[1], nodes[0])] = FALSIFY_VIOLATION_COLOR
            elif method == "validate_causal_minimality":
                # For causal minimality we highlight the edge Y \in Pa_X -> X
                colors[(nodes[1], nodes[0])] = FALSIFY_VIOLATION_COLOR

    plot(causal_graph, colors=colors)


def _generate_table(
    validation_repr, suggestion_repr, width=105, validation_name="Test Summary", suggestion_name="Suggestions"
):
    # Create Validation header
    _repr = [
        "+" + "-" * (width - 2) + "+\n",
        "|" + " " * int(np.floor((width - 2 - len(validation_name)) / 2)),
        validation_name,
        " " * int(np.ceil((width - 2 - len(validation_name)) / 2)) + "|\n",
        "+" + "-" * (width - 2) + "+\n",
    ]
    # Create Validation summary
    _repr += [f"| {v_line}" + " " * (width - 3 - len(v_line)) + "|\n" for v_line in validation_repr]
    # Close Validation
    _repr += ["+" + "-" * (width - 2) + "+\n"]
    # Create Suggestions header
    if suggestion_repr:
        _repr += [
            "|" + " " * int(np.floor((width - 2 - len(suggestion_name)) / 2)),
            suggestion_name,
            " " * int(np.ceil((width - 2 - len(suggestion_name)) / 2)) + "|\n",
            "+" + "-" * (width - 2) + "+\n",
        ]
    # Iterate over suggestions
    for m, suggestions in suggestion_repr.items():
        left_col = "| " + m + " |"
        if not suggestions:
            right_col = " " + " " * (width - 2 - len(left_col)) + "|\n"
            _repr += [left_col, right_col]
        for i, s in enumerate(suggestions):
            if i > 0:
                left_col = "| " + " " * len(m) + " |"
            right_col = " - " + s + " " * (width - 4 - len(s) - len(left_col)) + "|\n"
            _repr += [left_col, right_col]
        _repr += ["+" + "-" * (width - 2) + "+\n"]

    return "".join(_repr)[:-1]


def _compute_p_value(
    data: pd.DataFrame,
    X: Union[List, str],
    Y: Union[List, str],
    Z: Optional[Union[Set, List, str]],
    independence_test: Optional[Callable[[np.ndarray, np.ndarray], float]],
    conditional_independence_test: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    seed: int,
) -> float:
    """Perform (conditional) independence test and report p-value.

    :param data: Observations of variables in the DAG.
    :param X: Variable to test (conditional) independence with Y
    :param Y: Variable to test (conditional) independence with X
    :param Z: Set to condition independence test on. Can be empty (None, empty set, or empty list).
    :param independence_test: independence test to use.
    :param conditional_independence_test: Conditional independence test to use.
    :param seed: Random seed
    :return: p-value
    """
    set_random_seed(seed)

    if Z:
        p_value = conditional_independence_test(data[X].values, data[Y].values, data[Z].values)
    else:
        p_value = independence_test(data[X].values, data[Y].values)
    return p_value


def _get_parental_triples(causal_graph: DirectedGraph, include_unconditional: bool):
    """
    For a given graph collect all parental triples, that is, the triple (X, Y, Z) is a parental triple iff
        Y is non-descendant of X, and
        Z are the parents of X (can be empty if include_unconditional=True)
    """
    triples = []
    for node in causal_graph.nodes:
        parents = get_ordered_predecessors(causal_graph, node)
        non_descendants = _get_non_descendants(causal_graph, node, exclude_parents=True)
        if (parents or include_unconditional) and non_descendants:
            for non_desc in non_descendants:
                triples.append((node, non_desc, parents))
    return triples


def _permutation_based(
    causal_graph: DirectedGraph,
    data: pd.DataFrame,
    p_values_memory: _PValuesMemory,
    exclude_original_order: bool,
    n_permutations: int,
    methods: Union[Callable, Tuple[Callable, ...], List[Callable]],
    independence_test: Callable[[np.ndarray, np.ndarray], float],
    conditional_independence_test: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    significance_level: float,
    show_progress_bar: bool,
    **method_kwargs,
) -> Dict[str, List[Union[DirectedGraph, Dict]]]:
    """
    Generate baseline for node permutations.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param p_values_memory: _PValuesMemory object, where results of previously performed tests are stored.
    :param exclude_original_order: Exclude the original ordering of the nodes (default=False)
    :param n_permutations: Number of permutations to perform. If -1 use all n_nodes! - int(exclude_orig) permutations
    :param methods: Validation methods to perform. Supported are: validate_lmc, validate_ed.
    :param independence_test: Independence test to use for checking edge dependencies.
    :param conditional_independence_test: Conditional independence test to use for checking local Markov condition.
    :param significance_level: Significance level for (conditional) independence tests.
    :param disable_progress_bar: Disable the progress bar
    :return: Dictionary containing summary of validation for each individual graph as well as the permuted graphs
    """

    if not isinstance(methods, (tuple, list)):
        methods = (methods,)

    perm_gen = _PermuteNodes(causal_graph, n_permutations=n_permutations, exclude_original_order=exclude_original_order)
    validation_summary = {"permuted_graphs": [], **{m.__name__: [] for m in methods}}
    for permuted_graph in tqdm(perm_gen, desc="Test permutations of given graph", disable=not show_progress_bar):
        res = validate_graph(
            causal_graph=permuted_graph,
            data=data,
            p_values_memory=p_values_memory,
            causal_graph_reference=causal_graph,
            methods=methods,
            independence_test=independence_test,
            conditional_independence_test=conditional_independence_test,
            significance_level=significance_level,
            **method_kwargs,
        )
        validation_summary["permuted_graphs"].append(permuted_graph)
        for m in methods:
            validation_summary[m.__name__].append(res[m.__name__])

    return validation_summary


class _PermuteNodes:
    def __init__(self, causal_graph: DirectedGraph, exclude_original_order: bool, n_permutations: int):
        """
        Randomly permute the nodes of a given causal graph while keeping the underlying graph structure the same.
        :param causal_graph: A directed acyclic graph (DAG).
        :param exclude_original_order: Do not return the original order.
        :param n_permutations: Return a generator with n_permutations permutations. If n_permutations = -1 (default),
                we return all n_nodes! - int(exclude_orig) permutations.
        :return: Copy of causal_graph with nodes randomly permuted.
        """
        self.causal_graph = causal_graph
        self.exclude_original_order = exclude_original_order
        self.n_permutations = n_permutations
        self.max_perms = np.math.factorial(self.causal_graph.number_of_nodes()) - int(self.exclude_original_order)

        if n_permutations == -1 or n_permutations > self.max_perms:
            self.it = self.iter_all_permutations()
            self.length = self.max_perms
            if self.length > 2**63 - 1:
                raise ValueError(
                    f"Too many permutations specified. Did you accidently set 'n_permutations'=-1 for a "
                    f"large (>20 nodes) graph? "
                    f"Given graph has {causal_graph.number_of_nodes()} nodes."
                )
        else:
            self.it = self.iter_random_permutations()
            self.length = self.n_permutations

    def iter_all_permutations(self):
        for i, perm in enumerate(permutations(self.causal_graph.nodes)):
            if self.exclude_original_order and i == 0:
                continue
            mapping = {node: perm[i] for i, node in enumerate(self.causal_graph.nodes)}
            yield nx.relabel_nodes(self.causal_graph, mapping, copy=True)

    def iter_random_permutations(self):
        for _ in range(self.n_permutations):
            if self.exclude_original_order:
                is_orig = True
                while is_orig:
                    perm = list(np.random.permutation(self.causal_graph.nodes))
                    if perm != list(self.causal_graph.nodes):
                        is_orig = False
            else:
                perm = list(np.random.permutation(self.causal_graph.nodes))

            mapping = {node: perm[i] for i, node in enumerate(self.causal_graph.nodes)}
            yield nx.relabel_nodes(self.causal_graph, mapping, copy=True)

    def __iter__(self):
        yield from self.it

    def __len__(self):
        return self.length


def _to_frozenset(x: Union[Set, List, str]):
    """Converts a set, list or string into a hashable frozenset"""
    assert (
        isinstance(x, Set) or isinstance(x, List) or isinstance(x, str)
    ), f"{x} must be list, set or str. Got {type(x)} instead!"

    if isinstance(x, str):
        return frozenset({x})
    return frozenset(x)
