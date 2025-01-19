import copy
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefutation, CausalRefuter, test_significance

logger = logging.getLogger(__name__)


class RandomCommonCause(CausalRefuter):
    """Refute an estimate by introducing a randomly generated confounder
    (that may have been unobserved).

    Supports additional parameters that can be specified in the refute_estimate() method. For joblib-related parameters (n_jobs, verbose), please refer to the joblib documentation for more details (https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html).

    :param num_simulations: The number of simulations to be run, which is ``CausalRefuter.DEFAULT_NUM_SIMULATIONS`` by default
    :type num_simulations: int, optional

    :param random_state: The seed value to be added if we wish to repeat the same random behavior. If we with to repeat the same behavior we push the same seed in the psuedo-random generator
    :type random_state: int, RandomState, optional

    :param n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all (this is the default).
    :type n_jobs: int, optional

    :param verbose: The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported. The default is 0.
    :type verbose: int, optional
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._random_state = kwargs.pop("random_state", None)

        self.logger = logging.getLogger(__name__)

    def refute_estimate(self, show_progress_bar=False):
        refute = refute_random_common_cause(
            self._data,
            self._target_estimand,
            self._estimate,
            self._num_simulations,
            self._random_state,
            show_progress_bar,
            self._n_jobs,
            self._verbose,
        )
        refute.add_refuter(self)
        return refute


def _refute_once(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    random_state: Optional[np.random.RandomState] = None,
):
    if random_state is None:
        new_data = data.assign(w_random=np.random.randn(data.shape[0]))
    else:
        new_data = data.assign(w_random=random_state.normal(size=data.shape[0]))

    new_estimator = estimate.estimator.get_new_estimator_object(target_estimand)
    new_estimator.fit(
        new_data,
        effect_modifier_names=estimate.estimator._effect_modifier_names,
        **new_estimator._fit_params if hasattr(new_estimator, "_fit_params") else {},
    )
    new_effect = new_estimator.estimate_effect(
        new_data,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        target_units=estimate.estimator._target_units,
    )
    return new_effect.value


def refute_random_common_cause(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    num_simulations: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    show_progress_bar: bool = False,
    n_jobs: int = 1,
    verbose: int = 0,
    **_,
) -> CausalRefutation:
    """Refute an estimate by introducing a randomly generated confounder
    (that may have been unobserved).

    :param data: pd.DataFrame: Data to run the refutation
    :param target_estimand: IdentifiedEstimand: Identified estimand to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param num_simulations: The number of simulations to be run, which defaults to ``CausalRefuter.DEFAULT_NUM_SIMULATIONS``
    :param random_state: The seed value to be added if we wish to repeat the same random behavior. If we want to repeat the same behavior we push the same seed in the psuedo-random generator.
    :param n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all (this is the default).
    :param verbose: The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported. The default is 0.

    """
    logger.info("Refutation over {} simulated datasets, each with a random common cause added".format(num_simulations))

    new_adjustment_variables = target_estimand.get_adjustment_set() + ["w_random"]
    identified_estimand = copy.deepcopy(target_estimand)
    # Adding a new backdoor variable to the identified estimand
    identified_estimand.set_adjustment_set(new_adjustment_variables)

    if isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state)

    # Run refutation in parallel
    sample_estimates = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_refute_once)(data, identified_estimand, estimate, random_state)
        for _ in tqdm(
            range(num_simulations),
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            disable=not show_progress_bar,
            desc="Refuting Estimates: ",
        )
    )
    sample_estimates = np.array(sample_estimates)

    refute = CausalRefutation(
        estimate.value, np.mean(sample_estimates), refutation_type="Refute: Add a random common cause"
    )

    # We want to see if the estimate falls in the same distribution as the one generated by the refuter
    # Ideally that should be the case as choosing a subset should not have a significant effect on the ability
    # of the treatment to affect the outcome
    refute.add_significance_test_results(test_significance(estimate, sample_estimates))

    return refute
