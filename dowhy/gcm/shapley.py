"""This module provides functionality for shapley value estimation.

Classes and functions in this module should be considered experimental, meaning there might be breaking API changes in
the future.
"""

import itertools
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.special import comb
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import dowhy.gcm.config as config
from dowhy.gcm.constant import EPS
from dowhy.gcm.util.general import set_random_seed


class ShapleyApproximationMethods(Enum):
    """
    AUTO: Using EXACT when number of players is below 6 and EARLY_STOPPING otherwise.
    EXACT: Generate all possible subsets and estimate Shapley values with corresponding subset weights.
    EXACT_FAST: Generate all possible subsets and estimate Shapley values via weighed least squares regression. This can
                be faster, but, depending on the set function, numerically less stable.
    SUBSET_SAMPLING: Randomly samples subsets and estimate Shapley values via weighed least squares regression. Here,
                     only a certain number of randomly drawn subsets are used.
    EARLY_STOPPING: Estimate Shapley values based on a few randomly generated permutations. Stop the estimation process
                    when the the Shapley values do not change much on average anymore between runs.
    PERMUTATION: Estimates Shapley values based on a fixed number of randomly generated permutations. By fine tuning
                 hyperparameters, this can be potentially faster than the early stopping approach due to a better
                 utilization of the parallelization.
    """

    AUTO = (0,)
    EXACT = (1,)
    EXACT_FAST = (2,)
    EARLY_STOPPING = (3,)
    PERMUTATION = (4,)
    SUBSET_SAMPLING = 5


class ShapleyConfig:
    def __init__(
        self,
        approximation_method: ShapleyApproximationMethods = ShapleyApproximationMethods.AUTO,
        num_samples: int = 5000,
        min_percentage_change_threshold: float = 0.01,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Config for estimating Shapley values.

        :param approximation_method: Type of approximation methods (see :py:class:`ShapleyApproximationMethods <dowhy.gcm.shapley.ShapleyApproximationMethods>`).
        :param num_samples: Number of samples used for approximating the Shapley values. Depending on the approximation
                            method, this can either represent the number of drawn subsets (in SUBSET_SAMPLING) or the
                            number of drawn permutations (in EARLY_STOPPING and PERMUTATION). In case of EARLY_STOPPING,
                            this also represents a limit on the evaluation runs.
        :param min_percentage_change_threshold: This parameter is only relevant for EARLY_STOPPING and indicates the
                                                minimum required change of the Shapley values between two runs
                                                (i.e. evaluation of permutations) before the estimation stops.
                                                For instance, if the Shapley value changes less than the given value for
                                                a certain number of consecutive runs, the algorithm stops and returns
                                                the current result.
        :param n_jobs: Number of parallel jobs.
        """
        self.approximation_method = approximation_method
        self.num_samples = num_samples
        self.min_percentage_change_threshold = min_percentage_change_threshold
        self.n_jobs = config.default_n_jobs if n_jobs is None else n_jobs


def estimate_shapley_values(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    num_players: int,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Estimates the Shapley values based on the provided set function. A set function here is defined by taking a
    (subset) of players and returning a certain utility value. This is in the context of attributing the
    value of the i-th player to a subset of players S by evaluating v(S u {i}) - v(S), where v is the
    set function and i is not in S. While we use the term 'player' here, this is often a certain feature/variable.

    The input of the set function is a binary vector indicating which player is part of the set. For instance, given 4
    players (1,2,3,4) and a subset only contains players 1,2,4, then this is indicated by the vector [1, 1, 0, 1]. The
    function is expected to return a numeric value based on this input.

    Note: The set function can be arbitrary and can resemble computationally complex operations. Keep in mind
    that the estimation of Shapley values can become computationally expensive and requires a lot of memory. If the
    runtime is too slow, consider changing the default config.

    :param set_func: A set function that expects a binary vector as input which specifies which player is part of the
                     subset.
    :param num_players: Total number of players.
    :param shapley_config: A config object for indicating the approximation method and other parameters. If None is
                           given, a default config is used. For faster runtime or more accurate results, consider
                           creating a custom config.
    :return: A numpy array representing the Shapley values for each player, i.e. there are as many Shapley values as
             num_players. The i-th entry belongs to the i-th player. Here, the set function defines which index belongs
             to which player and is responsible to keep it consistent.
    """
    if shapley_config is None:
        shapley_config = ShapleyConfig()

    approximation_method = shapley_config.approximation_method
    if approximation_method == ShapleyApproximationMethods.AUTO:
        if num_players <= 5:
            approximation_method = ShapleyApproximationMethods.EXACT
        else:
            approximation_method = ShapleyApproximationMethods.EARLY_STOPPING

    if approximation_method == ShapleyApproximationMethods.EXACT:
        return _estimate_shapley_values_exact(set_func=set_func, num_players=num_players, n_jobs=shapley_config.n_jobs)
    elif approximation_method == ShapleyApproximationMethods.PERMUTATION:
        return _approximate_shapley_values_via_permutation_sampling(
            set_func=set_func,
            num_players=num_players,
            num_permutations=max(1, shapley_config.num_samples // num_players),
            n_jobs=shapley_config.n_jobs,
        )
    elif approximation_method == ShapleyApproximationMethods.EARLY_STOPPING:
        return _approximate_shapley_values_via_early_stopping(
            set_func=set_func,
            num_players=num_players,
            max_runs=shapley_config.num_samples,
            min_percentage_change_threshold=shapley_config.min_percentage_change_threshold,
            n_jobs=shapley_config.n_jobs,
        )
    elif approximation_method == ShapleyApproximationMethods.SUBSET_SAMPLING:
        return _approximate_shapley_values_via_least_squares_regression(
            set_func=set_func,
            num_players=num_players,
            use_subset_approximation=True,
            num_samples_for_approximation=shapley_config.num_samples,
            n_jobs=shapley_config.n_jobs,
        )
    elif approximation_method == ShapleyApproximationMethods.EXACT_FAST:
        return _approximate_shapley_values_via_least_squares_regression(
            set_func=set_func,
            num_players=num_players,
            use_subset_approximation=False,
            num_samples_for_approximation=shapley_config.num_samples,
            n_jobs=shapley_config.n_jobs,
        )
    else:
        raise ValueError("Unknown method for Shapley approximation!")


def _estimate_shapley_values_exact(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]], num_players: int, n_jobs: int
) -> np.ndarray:
    """Following Eq. (2) in
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR."""
    all_subsets = [tuple(subset) for subset in itertools.product([0, 1], repeat=num_players)]

    with Parallel(n_jobs=n_jobs) as parallel:
        subset_to_result_map = _evaluate_set_function(set_func, all_subsets, parallel)

    def compute_subset_weight(length: int) -> float:
        return 1 / (num_players * comb(num_players - 1, length))

    subset_weight_cache = {}

    shapley_values = [None] * num_players
    subsets_missing_one_player = np.array(list(itertools.product([0, 1], repeat=num_players - 1)))
    for player_index in range(num_players):
        subsets_with_player = [
            tuple(subset) for subset in np.insert(subsets_missing_one_player, player_index, 1, axis=1)
        ]
        subsets_without_player = [
            tuple(subset) for subset in np.insert(subsets_missing_one_player, player_index, 0, axis=1)
        ]

        for i in range(len(subsets_with_player)):
            subset_length = int(np.sum(subsets_without_player[i]))
            if subset_length not in subset_weight_cache:
                subset_weight_cache[subset_length] = compute_subset_weight(subset_length)

            weighted_diff = subset_weight_cache[subset_length] * (
                subset_to_result_map[subsets_with_player[i]] - subset_to_result_map[subsets_without_player[i]]
            )
            # For estimating Shapley values for multiple samples (e.g. in feature relevance) and the number of samples
            # is unknown beforehand.
            if shapley_values[player_index] is None:
                shapley_values[player_index] = weighted_diff
            else:
                shapley_values[player_index] += weighted_diff

    return np.array(shapley_values).T


def _approximate_shapley_values_via_least_squares_regression(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    num_players: int,
    use_subset_approximation: bool,
    num_samples_for_approximation: int,
    n_jobs: int,
    full_and_empty_subset_weight: float = 10**20,
) -> np.ndarray:
    """For more details about this approximation, see Section 4.1.1 in
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR."""
    if not use_subset_approximation:
        all_subsets, weights = _create_subsets_and_weights_exact(num_players, full_and_empty_subset_weight)
    else:
        all_subsets, weights = _create_subsets_and_weights_approximation(
            num_players, full_and_empty_subset_weight, num_samples_for_approximation
        )

    def parallel_job(subset: np.ndarray, parallel_random_seed: int):
        set_random_seed(parallel_random_seed)

        return set_func(subset)

    with Parallel(n_jobs=n_jobs) as parallel:
        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(all_subsets))
        set_function_results = parallel(
            delayed(parallel_job)(subset, random_seed)
            for subset, random_seed in tqdm(
                zip(all_subsets, random_seeds),
                desc="Estimate shapley values as least squares solution",
                position=0,
                leave=True,
                disable=not config.show_progress_bars,
            )
        )

    return LinearRegression().fit(all_subsets, np.array(set_function_results), sample_weight=weights).coef_


def _approximate_shapley_values_via_permutation_sampling(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]], num_players: int, num_permutations: int, n_jobs: int
) -> np.ndarray:
    """For more details about this approximation, see
    Strumbelj, E., Kononenko, I. (2014).
    Explaining prediction models and individual predictions with feature contributions.
    In Knowledge and information systems, 41(3):647–665"""
    full_subset_result, empty_subset_result = _estimate_full_and_emtpy_subset_results(set_func, num_players)

    subsets_to_evaluate = set()
    all_permutations = []
    for i in range(num_permutations):
        permutation = np.random.choice(num_players, num_players, replace=False)
        all_permutations.append(permutation)

        subsets_to_evaluate.update(_create_index_order_and_subset_tuples(permutation))

    with Parallel(n_jobs=n_jobs) as parallel:
        evaluated_subsets = _evaluate_set_function(set_func, subsets_to_evaluate, parallel)

    shapley_values = _estimate_shapley_values_of_permutation(
        all_permutations[0], evaluated_subsets, full_subset_result, empty_subset_result
    )
    for i in range(1, len(all_permutations)):
        shapley_values += _estimate_shapley_values_of_permutation(
            all_permutations[i], evaluated_subsets, full_subset_result, empty_subset_result
        )

    return shapley_values / len(all_permutations)


def _approximate_shapley_values_via_early_stopping(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    num_players: int,
    max_runs: int,
    min_percentage_change_threshold: float,
    n_jobs: int,
    num_permutations_per_run: int = 5,
) -> np.ndarray:
    """Combines the approximation method described in

    Strumbelj, E., Kononenko, I. (2014).
    Explaining prediction models and individual predictions with feature contributions.
    In Knowledge and information systems, 41(3):647–665

    with an early stopping criteria. This is, if the Shapley values change less than a certain threshold on average
    between two runs, then stop the estimation.
    """
    full_subset_result, empty_subset_result = _estimate_full_and_emtpy_subset_results(set_func, num_players)

    shapley_values = None
    old_shap_proxy = np.zeros(num_players)
    evaluated_subsets = {}
    num_generated_permutations = 0
    run_counter = 0
    converged_run = 0

    if config.show_progress_bars:
        pbar = tqdm(total=1)

    with Parallel(n_jobs=n_jobs) as parallel:
        # The method stops if either the change between some consecutive runs is below the given threshold or the
        # maximum number of runs is reached.
        while True:
            run_counter += 1
            subsets_to_evaluate = set()

            # In each run, we create one random permutation of players. For instance, given 4 players, a permutation
            # could be [3,1,4,2].
            permutations = [
                np.random.choice(num_players, num_players, replace=False) for _ in range(num_permutations_per_run)
            ]
            for permutation in permutations:
                num_generated_permutations += 1
                # Create all subsets belonging to the generated permutation. This is, if we have [3,1,4,2], then the
                # subsets are [3], [3,1], [3,1,4] [3,1,4,2].
                subsets_to_evaluate.update(
                    [
                        subset_tuple
                        for subset_tuple in _create_index_order_and_subset_tuples(permutation)
                        if subset_tuple not in evaluated_subsets
                    ]
                )

            # The result for each subset is cached such that if a subset that has already been evaluated appears again,
            # we can take this result directly.
            evaluated_subsets.update(_evaluate_set_function(set_func, subsets_to_evaluate, parallel, False))

            for permutation in permutations:
                # To improve the runtime, multiple permutations are evaluated in each run.
                if shapley_values is None:
                    shapley_values = _estimate_shapley_values_of_permutation(
                        permutation, evaluated_subsets, full_subset_result, empty_subset_result
                    )
                else:
                    shapley_values += _estimate_shapley_values_of_permutation(
                        permutation, evaluated_subsets, full_subset_result, empty_subset_result
                    )

            if run_counter > max_runs:
                break

            new_shap_proxy = np.array(shapley_values)
            new_shap_proxy[new_shap_proxy == 0] = EPS
            # The current Shapley values are the average of the estimated values, i.e. we need to divide by the number
            # of generated permutations here.
            new_shap_proxy /= num_generated_permutations

            if run_counter > 1:
                percentage_changes = 1 - new_shap_proxy / old_shap_proxy
                if config.show_progress_bars:
                    pbar.set_description(
                        f"Estimating Shapley Values. "
                        f"Average change of Shapley values in run {run_counter} "
                        f"({num_generated_permutations} evaluated permutations): "
                        f"{np.mean(percentage_changes) * 100}%"
                    )

                if np.mean(percentage_changes) < min_percentage_change_threshold:
                    # Here, the change between two runs is below the minimum threshold, but to reduce the likelihood
                    # that this just happened by chance, we require that this happens at least for two runs in a row.
                    converged_run += 1
                    if converged_run >= 2:
                        break
                else:
                    converged_run = 0

            old_shap_proxy = new_shap_proxy

    if config.show_progress_bars:
        pbar.update(1)
        pbar.close()

    return shapley_values / num_generated_permutations


def _create_subsets_and_weights_exact(num_players: int, high_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """Creates all subsets and the exact weights of each subset. See Section 4.1.1. in

    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR.

    for more details on this.

    :param num_players: Total number of players.
    :param high_weight: A 'high' weight for computational purposes. This is used to resemble 'infinity', but needs to be
                        selected carefully to avoid numerical issues.
    :return: A tuple, where the first entry is a numpy array with all subsets and the second entry is an array with the
             corresponding weights to each subset.
    """
    all_subsets = []

    num_iterations = int(np.ceil(num_players / 2))

    for i in range(num_iterations):
        # Create all (unique) subsets)
        all_subsets.extend(
            np.array(
                [np.bincount(combs, minlength=num_players) for combs in itertools.combinations(range(num_players), i)]
            )
        )

        all_subsets.extend(
            np.array(
                [
                    np.bincount(combs, minlength=num_players)
                    for combs in itertools.combinations(range(num_players), num_players - i)
                ]
            )
        )

        if i == num_iterations - 1 and num_players % 2 == 0:
            all_subsets.extend(
                np.array(
                    [
                        np.bincount(combs, minlength=num_players)
                        for combs in itertools.combinations(range(num_players), i + 1)
                    ]
                )
            )

    weights = np.zeros(len(all_subsets))

    for i, subset in enumerate(all_subsets):
        subset_size = np.sum(subset)
        if subset_size == num_players or subset_size == 0:
            # Assigning a 'high' weight, since this resembles "infinity".
            weights[i] = high_weight
        else:
            # The weight for a subset with a specific length (see paper mentioned in the docstring for more
            # information).
            weights[i] = (num_players - 1) / (
                scipy.special.binom(num_players, subset_size) * subset_size * (num_players - subset_size)
            )

    return np.array(all_subsets, dtype=np.int), weights.astype(np.float)


def _create_subsets_and_weights_approximation(
    num_players: int, high_weight: float, num_subset_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly samples subsets and weights them based on the number of how often they appear.

    :param num_players: Total number of players.
    :param high_weight: A 'high' weight for computational purposes. This is used to resemble 'infinity', but needs to be
                        selected carefully to avoid numerical issues.
    :param num_subset_samples: Number of subset samples.
    :return: A tuple, where the first entry is a numpy array with the sampled subsets and the second entry is an array
             with the corresponding weights to each subset.
    """
    all_subsets = [np.zeros(num_players), np.ones(num_players)]
    weights = {tuple(all_subsets[0]): high_weight, tuple(all_subsets[1]): high_weight}

    probabilities_of_subset_length = np.zeros(num_players + 1)
    for i in range(1, num_players):
        probabilities_of_subset_length[i] = (num_players - 1) / (i * (num_players - i))

    probabilities_of_subset_length = probabilities_of_subset_length / np.sum(probabilities_of_subset_length)

    for i in range(num_subset_samples):
        subset_as_tuple = _convert_list_of_indices_to_binary_vector_as_tuple(
            np.random.choice(
                num_players, np.random.choice(num_players + 1, 1, p=probabilities_of_subset_length), replace=False
            ),
            num_players,
        )

        if subset_as_tuple not in weights:
            weights[subset_as_tuple] = 0
            all_subsets.append(np.array(subset_as_tuple))

        weights[subset_as_tuple] += 1

    weights = np.array([weights[tuple(x)] for x in all_subsets])

    return np.array(all_subsets, dtype=np.int), weights.astype(np.float)


def _convert_list_of_indices_to_binary_vector_as_tuple(list_of_indices: List[int], num_players: int) -> Tuple[int]:
    subset = np.zeros(num_players, dtype=np.int)
    subset[list_of_indices] = 1

    return tuple(subset)


def _evaluate_set_function(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    evaluation_subsets: Union[Set[Tuple[int]], List[Tuple[int]]],
    parallel_context: Parallel,
    show_progressbar: bool = True,
) -> Dict[Tuple[int], Union[float, np.ndarray]]:
    def parallel_job(input_subset: Tuple[int], parallel_random_seed: int) -> Union[float, np.ndarray]:
        set_random_seed(parallel_random_seed)

        return set_func(np.array(input_subset))

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(evaluation_subsets))
    subset_results = parallel_context(
        delayed(parallel_job)(subset_to_evaluate, random_seed)
        for subset_to_evaluate, random_seed in tqdm(
            zip(evaluation_subsets, random_seeds),
            desc="Evaluate set function",
            position=0,
            leave=True,
            disable=not config.show_progress_bars or not show_progressbar,
        )
    )

    subset_to_result_map = {}
    for (subset, result) in zip(evaluation_subsets, subset_results):
        subset_to_result_map[subset] = result

    return subset_to_result_map


def _estimate_full_and_emtpy_subset_results(
    set_func: Callable[[np.ndarray], Union[float, np.ndarray]], num_players: int
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    return set_func(np.ones(num_players, dtype=np.int)), set_func(np.zeros(num_players, dtype=np.int))


def _create_index_order_and_subset_tuples(permutation: List[int]) -> List[Tuple[int]]:
    indices = []
    index_tuples = []

    for var in range(len(permutation) - 1):
        indices += [permutation[var]]
        index_tuples.append(_convert_list_of_indices_to_binary_vector_as_tuple(indices, len(permutation)))

    return index_tuples


def _estimate_shapley_values_of_permutation(
    permutation: List[int],
    evaluated_subsets: Dict[Tuple[int], Union[float, np.ndarray]],
    full_subset_result: Union[float, np.ndarray],
    empty_subset_result: Union[float, np.ndarray],
) -> np.ndarray:
    current_variable_set = []
    shapley_values = [[]] * len(permutation)
    previous_result = empty_subset_result
    for n in range(len(permutation) - 1):
        current_variable_set += [permutation[n]]
        current_result = evaluated_subsets[
            _convert_list_of_indices_to_binary_vector_as_tuple(current_variable_set, len(permutation))
        ]

        shapley_values[permutation[n]] = current_result - previous_result
        previous_result = current_result

    shapley_values[permutation[-1]] = full_subset_result - previous_result

    return np.array(shapley_values).T
