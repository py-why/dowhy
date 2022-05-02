import itertools
from enum import Enum
from typing import Callable, Union, Tuple, List, Dict, Set, Optional

import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.special import comb
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import dowhy.gcm.config as config
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
    AUTO = 0,
    EXACT = 1,
    EXACT_FAST = 2,
    EARLY_STOPPING = 3,
    PERMUTATION = 4,
    SUBSET_SAMPLING = 5


class ShapleyConfig:
    def __init__(self,
                 approximation_method: ShapleyApproximationMethods = ShapleyApproximationMethods.AUTO,
                 num_samples: int = 5000,
                 min_percentage_change_threshold: float = 0.01,
                 n_jobs: Optional[int] = None) -> None:
        self.approximation_method = approximation_method
        self.num_samples = num_samples
        self.min_percentage_change_threshold = min_percentage_change_threshold
        self.n_jobs = config.default_n_jobs if n_jobs is None else n_jobs


def estimate_shapley_values(set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
                            num_players: int,
                            shapley_config: Optional[ShapleyConfig] = None) -> np.ndarray:
    if shapley_config is None:
        shapley_config = ShapleyConfig()

    approximation_method = shapley_config.approximation_method
    if approximation_method == ShapleyApproximationMethods.AUTO:
        if num_players <= 5:
            approximation_method = ShapleyApproximationMethods.EXACT
        else:
            approximation_method = ShapleyApproximationMethods.EARLY_STOPPING

    if approximation_method == ShapleyApproximationMethods.EXACT:
        return _estimate_shapley_values_exact(set_func=set_func,
                                              num_players=num_players,
                                              n_jobs=shapley_config.n_jobs)
    elif approximation_method == ShapleyApproximationMethods.PERMUTATION:
        return _approximate_shapley_values_via_permutation_sampling(
            set_func=set_func,
            num_players=num_players,
            num_permutations=max(1, shapley_config.num_samples // num_players),
            n_jobs=shapley_config.n_jobs)
    elif approximation_method == ShapleyApproximationMethods.EARLY_STOPPING:
        return _approximate_shapley_values_via_early_stopping(
            set_func=set_func,
            num_players=num_players,
            max_runs=shapley_config.num_samples,
            min_percentage_change_threshold=shapley_config.min_percentage_change_threshold,
            n_jobs=shapley_config.n_jobs)
    elif approximation_method == ShapleyApproximationMethods.SUBSET_SAMPLING:
        return _approximate_shapley_values_via_least_squares_regression(
            set_func=set_func,
            num_players=num_players,
            use_subset_approximation=True,
            num_samples_for_approximation=shapley_config.num_samples,
            n_jobs=shapley_config.n_jobs)
    elif approximation_method == ShapleyApproximationMethods.EXACT_FAST:
        return _approximate_shapley_values_via_least_squares_regression(
            set_func=set_func,
            num_players=num_players,
            use_subset_approximation=False,
            num_samples_for_approximation=shapley_config.num_samples,
            n_jobs=shapley_config.n_jobs)
    else:
        raise ValueError("Unknown method for Shapley approximation!")


def _estimate_shapley_values_exact(set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
                                   num_players: int,
                                   n_jobs: int) -> np.ndarray:
    """ Following Eq. (2) in
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR. """
    all_subsets = [tuple(subset) for subset in itertools.product([0, 1], repeat=num_players)]

    with Parallel(n_jobs=n_jobs) as parallel:
        subset_to_result_map = _evaluate_set_function(set_func, all_subsets, parallel)

    def compute_subset_weight(length: int) -> float:
        return 1 / (num_players * comb(num_players - 1, length))

    subset_weight_cache = {}

    shapley_values = [None] * num_players
    subsets_missing_one_player = np.array(list(itertools.product([0, 1], repeat=num_players - 1)))
    for player_index in range(num_players):
        subsets_with_player = [tuple(subset)
                               for subset in np.insert(subsets_missing_one_player, player_index, 1, axis=1)]
        subsets_without_player = [tuple(subset)
                                  for subset in np.insert(subsets_missing_one_player, player_index, 0, axis=1)]

        for i in range(len(subsets_with_player)):
            subset_length = int(np.sum(subsets_without_player[i]))
            if subset_length not in subset_weight_cache:
                subset_weight_cache[subset_length] = compute_subset_weight(subset_length)

            weighted_diff = \
                subset_weight_cache[subset_length] * (subset_to_result_map[subsets_with_player[i]]
                                                      - subset_to_result_map[subsets_without_player[i]])
            # For estimating Shapley values for multiple samples (e.g. in feature relevance) and the number of samples
            # is unknown beforehand.
            if shapley_values[player_index] is None:
                shapley_values[player_index] = weighted_diff
            else:
                shapley_values[player_index] += weighted_diff

    return np.array(shapley_values).T


def _approximate_shapley_values_via_least_squares_regression(set_func: Callable[[np.ndarray],
                                                                                Union[float, np.ndarray]],
                                                             num_players: int,
                                                             use_subset_approximation: bool,
                                                             num_samples_for_approximation: int,
                                                             n_jobs: int,
                                                             full_and_empty_subset_weight: float = 10 ** 20) \
        -> np.ndarray:
    """ For more details about this approximation, see Section 4.1.1 in
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR. """
    if not use_subset_approximation:
        all_subsets, weights = _create_subsets_and_weights_exact(num_players,
                                                                 full_and_empty_subset_weight)
    else:
        all_subsets, weights = _create_subsets_and_weights_approximation(num_players,
                                                                         full_and_empty_subset_weight,
                                                                         num_samples_for_approximation)

    def parallel_job(subset: np.ndarray, parallel_random_seed: int):
        set_random_seed(parallel_random_seed)

        return set_func(subset)

    with Parallel(n_jobs=n_jobs) as parallel:
        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(all_subsets))
        set_function_results = parallel(delayed(parallel_job)(subset, random_seed)
                                        for subset, random_seed in
                                        tqdm(zip(all_subsets, random_seeds),
                                             desc="Estimate shapley values as least squares solution",
                                             position=0, leave=True, disable=not config.show_progress_bars))

    return LinearRegression().fit(all_subsets, np.array(set_function_results), sample_weight=weights).coef_


def _approximate_shapley_values_via_permutation_sampling(
        set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
        num_players: int,
        num_permutations: int,
        n_jobs: int) -> np.ndarray:
    """ For more details about this approximation, see
    Strumbelj, E., Kononenko, I. (2014).
    Explaining prediction models and individual predictions with feature contributions.
    In Knowledge and information systems, 41(3):647–665 """
    full_subset_result, empty_subset_result = _estimate_full_and_emtpy_subset_results(set_func, num_players)

    subsets_to_evaluate = set()
    all_permutations = []
    for i in range(num_permutations):
        permutation = np.random.choice(num_players, num_players, replace=False)
        all_permutations.append(permutation)

        subsets_to_evaluate.update(_create_index_order_and_subset_tuples(permutation))

    with Parallel(n_jobs=n_jobs) as parallel:
        evaluated_subsets = _evaluate_set_function(set_func, subsets_to_evaluate, parallel)

    shapley_values = _estimate_shapley_values_of_permutation(all_permutations[0], evaluated_subsets,
                                                             full_subset_result, empty_subset_result)
    for i in range(1, len(all_permutations)):
        shapley_values += _estimate_shapley_values_of_permutation(all_permutations[i], evaluated_subsets,
                                                                  full_subset_result, empty_subset_result)

    return shapley_values / len(all_permutations)


def _approximate_shapley_values_via_early_stopping(
        set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
        num_players: int,
        max_runs: int,
        min_percentage_change_threshold: float,
        n_jobs: int,
        num_permutations_per_run: int = 5) -> np.ndarray:
    """ Combines the approximation method described in

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
        while True:
            run_counter += 1
            subsets_to_evaluate = set()

            permutations = [np.random.choice(num_players, num_players, replace=False)
                            for _ in range(num_permutations_per_run)]
            for permutation in permutations:
                num_generated_permutations += 1
                subsets_to_evaluate.update([subset_tuple for subset_tuple
                                            in _create_index_order_and_subset_tuples(permutation)
                                            if subset_tuple not in evaluated_subsets])

            evaluated_subsets.update(_evaluate_set_function(set_func,
                                                            subsets_to_evaluate,
                                                            parallel,
                                                            False))

            for permutation in permutations:
                if shapley_values is None:
                    shapley_values = _estimate_shapley_values_of_permutation(permutation, evaluated_subsets,
                                                                             full_subset_result, empty_subset_result)
                else:
                    shapley_values += _estimate_shapley_values_of_permutation(permutation, evaluated_subsets,
                                                                              full_subset_result, empty_subset_result)

            if run_counter > max_runs:
                break

            new_shap_proxy = np.array(shapley_values)
            new_shap_proxy[new_shap_proxy == 0] = config.EPS
            new_shap_proxy /= num_generated_permutations

            if run_counter > 1:
                percentage_changes = 1 - new_shap_proxy / old_shap_proxy
                if config.show_progress_bars:
                    pbar.set_description(f'Estimating Shapley Values. '
                                         f'Average change of Shapley values in run {run_counter} '
                                         f'({num_generated_permutations} evaluated permutations): '
                                         f'{np.mean(percentage_changes) * 100}%')

                if np.mean(percentage_changes) < min_percentage_change_threshold:
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
    all_subsets = []

    num_iterations = int(np.ceil(num_players / 2))

    for i in range(num_iterations):
        all_subsets.extend(np.array([np.bincount(combs, minlength=num_players) for combs in
                                     itertools.combinations(range(num_players), i)]))

        all_subsets.extend(np.array([np.bincount(combs, minlength=num_players) for combs in
                                     itertools.combinations(range(num_players), num_players - i)]))

        if i == num_iterations - 1 and num_players % 2 == 0:
            all_subsets.extend(np.array([np.bincount(combs, minlength=num_players) for combs in
                                         itertools.combinations(range(num_players), i + 1)]))

    weights = np.zeros(len(all_subsets))

    for i, subset in enumerate(all_subsets):
        subset_size = np.sum(subset)
        if subset_size == num_players or subset_size == 0:
            weights[i] = high_weight
        else:
            weights[i] = (num_players - 1) / (
                    scipy.special.binom(num_players, subset_size)
                    * subset_size
                    * (num_players - subset_size))

    return np.array(all_subsets, dtype=np.int), weights.astype(np.float)


def _create_subsets_and_weights_approximation(num_players: int, high_weight: float,
                                              num_subset_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    all_subsets = [np.zeros(num_players), np.ones(num_players)]
    weights = {tuple(all_subsets[0]): high_weight, tuple(all_subsets[1]): high_weight}

    probabilities_of_subset_length = np.zeros(num_players + 1)
    for i in range(1, num_players):
        probabilities_of_subset_length[i] = (num_players - 1) / (i * (num_players - i))

    probabilities_of_subset_length = probabilities_of_subset_length / np.sum(probabilities_of_subset_length)

    for i in range(num_subset_samples):
        subset_as_tuple = _convert_list_of_indices_to_binary_vector_as_tuple(
            np.random.choice(num_players,
                             np.random.choice(num_players + 1, 1, p=probabilities_of_subset_length),
                             replace=False), num_players)

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


def _evaluate_set_function(set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
                           evaluation_subsets: Union[Set[Tuple[int]], List[Tuple[int]]],
                           parallel_context: Parallel,
                           show_progressbar: bool = True) -> Dict[Tuple[int], Union[float, np.ndarray]]:
    def parallel_job(input_subset: Tuple[int], parallel_random_seed: int) -> Union[float, np.ndarray]:
        set_random_seed(parallel_random_seed)

        return set_func(np.array(input_subset))

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(evaluation_subsets))
    subset_results = parallel_context(delayed(parallel_job)(subset_to_evaluate, random_seed)
                                      for subset_to_evaluate, random_seed in
                                      tqdm(zip(evaluation_subsets, random_seeds),
                                           desc="Evaluate set function",
                                           position=0,
                                           leave=True, disable=not config.show_progress_bars or not show_progressbar))

    subset_to_result_map = {}
    for (subset, result) in zip(evaluation_subsets, subset_results):
        subset_to_result_map[subset] = result

    return subset_to_result_map


def _estimate_full_and_emtpy_subset_results(set_func: Callable[[np.ndarray], Union[float, np.ndarray]],
                                            num_players: int) \
        -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    return set_func(np.ones(num_players, dtype=np.int)), \
           set_func(np.zeros(num_players, dtype=np.int))


def _create_index_order_and_subset_tuples(permutation: List[int]) -> List[Tuple[int]]:
    indices = []
    index_tuples = []

    for var in range(len(permutation) - 1):
        indices += [permutation[var]]
        index_tuples.append(_convert_list_of_indices_to_binary_vector_as_tuple(indices, len(permutation)))

    return index_tuples


def _estimate_shapley_values_of_permutation(permutation: List[int],
                                            evaluated_subsets: Dict[Tuple[int], Union[float, np.ndarray]],
                                            full_subset_result: Union[float, np.ndarray],
                                            empty_subset_result: Union[float, np.ndarray]) -> np.ndarray:
    current_variable_set = []
    shapley_values = [[]] * len(permutation)
    previous_result = empty_subset_result
    for n in range(len(permutation) - 1):
        current_variable_set += [permutation[n]]
        current_result = evaluated_subsets[
            _convert_list_of_indices_to_binary_vector_as_tuple(current_variable_set, len(permutation))]

        shapley_values[permutation[n]] = current_result - previous_result
        previous_result = current_result

    shapley_values[permutation[-1]] = full_subset_result - previous_result

    return np.array(shapley_values).T
