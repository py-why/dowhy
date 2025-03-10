import logging
import random
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import resample
from tqdm.auto import tqdm

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefutation, CausalRefuter, choose_variables, test_significance

logger = logging.getLogger(__name__)


class BootstrapRefuter(CausalRefuter):
    """
    Refute an estimate by running it on a random sample of the data containing measurement error in the
    confounders. This allows us to find the ability of the estimator to find the effect of the
    treatment on the outcome.

    It supports additional parameters that can be specified in the refute_estimate() method.

    :param num_simulations: The number of simulations to be run, ``CausalRefuter.DEFAULT_NUM_SIMULATIONS`` by default
    :type num_simulations: int, optional

    :param sample_size: The size of each bootstrap sample and is the size of the original data by default
    :type sample_size: int, optional

    :param required_variables: The list of variables to be used as the input for ``y~f(W)``
      This is ``True`` by default, which in turn selects all variables leaving the treatment and the outcome
    :type required_variables: int, list, bool, optional

    1. An integer argument refers to how many variables will be used for estimating the value of the outcome
    2. A list explicitly refers to which variables will be used to estimate the outcome
       Furthermore, it gives the ability to explictly select or deselect the covariates present in the estimation of the
       outcome. This is done by either adding or explicitly removing variables from the list as shown below:

    .. note::
            * We need to pass required_variables = ``[W0,W1]`` if we want ``W0`` and ``W1``.
            * We need to pass required_variables = ``[-W0,-W1]`` if we want all variables excluding ``W0`` and ``W1``.

    3. If the value is True, we wish to include all variables to estimate the value of the outcome.

    .. warning:: A ``False`` value is ``INVALID`` and will result in an ``error``.

    :param noise: The standard deviation of the noise to be added to the data and is ``BootstrapRefuter.DEFAULT_STD_DEV`` by default
    :type noise: float, optional

    :param probability_of_change: It specifies the probability with which we change the data for a boolean or categorical variable
      It is ``noise`` by default, only if the value of ``noise`` is less than 1.
    :type probability_of_change: float, optional

    :param random_state: The seed value to be added if we wish to repeat the same random behavior. For this purpose, we repeat the same seed in the psuedo-random generator.
    :type random_state: int, RandomState, optional
    """

    DEFAULT_STD_DEV = 0.1
    DEFAULT_SUCCESS_PROBABILITY = 0.5
    DEFAULT_NUMBER_OF_TRIALS = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._sample_size = kwargs.pop("sample_size", len(self._data))
        self._required_variables = kwargs.pop("required_variables", True)
        self._noise = kwargs.pop("noise", BootstrapRefuter.DEFAULT_STD_DEV)
        self._probability_of_change = kwargs.pop("probability_of_change", None)
        self._random_state = kwargs.pop("random_state", None)

        self.logger = logging.getLogger(__name__)

    def refute_estimate(self, show_progress_bar: bool = False, *args, **kwargs):
        refute = refute_bootstrap(
            data=self._data,
            target_estimand=self._target_estimand,
            estimate=self._estimate,
            num_simulations=self._num_simulations,
            random_state=self._random_state,
            sample_size=self._sample_size,
            required_variables=self._required_variables,
            noise=self._noise,
            probability_of_change=self._probability_of_change,
            show_progress_bar=show_progress_bar,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )
        refute.add_refuter(self)
        return refute


def _refute_once(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    chosen_variables: Optional[List] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    sample_size: Optional[int] = None,
    noise: float = 0.1,
    probability_of_change: Optional[float] = None,
):
    if random_state is None:
        new_data = resample(data, n_samples=sample_size)
    else:
        new_data = resample(data, n_samples=sample_size, random_state=random_state)

    if chosen_variables is not None:
        for variable in chosen_variables:

            if ("float" or "int") in new_data[variable].dtype.name:
                scaling_factor = new_data[variable].std()
                new_data[variable] += np.random.normal(loc=0.0, scale=noise * scaling_factor, size=sample_size)

            elif "bool" in new_data[variable].dtype.name:
                probs = np.random.uniform(0, 1, sample_size)
                new_data[variable] = np.where(
                    probs < probability_of_change, np.logical_not(new_data[variable]), new_data[variable]
                )

            elif "category" in new_data[variable].dtype.name:
                categories = new_data[variable].unique()
                # Find the set difference for each row
                changed_data = new_data[variable].apply(lambda row: list(set(categories) - set([row])))
                # Choose one out of the remaining
                changed_data = changed_data.apply(lambda row: random.choice(row))
                new_data[variable] = np.where(probs < probability_of_change, changed_data)
                new_data[variable].astype("category")

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


def refute_bootstrap(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    num_simulations: int = 100,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    sample_size: Optional[int] = None,
    required_variables: bool = True,
    noise: float = 0.1,
    probability_of_change: Optional[float] = None,
    show_progress_bar: bool = False,
    n_jobs: int = 1,
    verbose: int = 0,
    **_,
) -> CausalRefutation:
    """Refute an estimate by running it on a random sample of the data containing measurement error in the
    confounders. This allows us to find the ability of the estimator to find the effect of the
    treatment on the outcome.

    :param data: pd.DataFrame: Data to run the refutation
    :param target_estimand: IdentifiedEstimand: Identified estimand to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param num_simulations: The number of simulations to be run, ``CausalRefuter.DEFAULT_NUM_SIMULATIONS`` by default
    :param random_state: The seed value to be added if we wish to repeat the same random behavior. For this purpose, we repeat the same seed in the psuedo-random generator.
    :param sample_size: The size of each bootstrap sample and is the size of the original data by default
    :param required_variables: The list of variables to be used as the input for ``y~f(W)``
      This is ``True`` by default, which in turn selects all variables leaving the treatment and the outcome
    1. An integer argument refers to how many variables will be used for estimating the value of the outcome
    2. A list explicitly refers to which variables will be used to estimate the outcome
       Furthermore, it gives the ability to explictly select or deselect the covariates present in the estimation of the
       outcome. This is done by either adding or explicitly removing variables from the list as shown below:
    .. note::
            * We need to pass required_variables = ``[W0,W1]`` if we want ``W0`` and ``W1``.
            * We need to pass required_variables = ``[-W0,-W1]`` if we want all variables excluding ``W0`` and ``W1``.
    3. If the value is True, we wish to include all variables to estimate the value of the outcome.
    .. warning:: A ``False`` value is ``INVALID`` and will result in an ``error``.
    :param noise: The standard deviation of the noise to be added to the data and is ``BootstrapRefuter.DEFAULT_STD_DEV`` by default
    :param probability_of_change: It specifies the probability with which we change the data for a boolean or categorical variable
      It is ``noise`` by default, only if the value of ``noise`` is less than 1.
    :param n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all (this is the default).
    :param verbose: The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported. The default is 0.
    """
    if sample_size is None:
        sample_size = len(data)

    chosen_variables = choose_variables(
        required_variables,
        target_estimand.get_adjustment_set()
        + target_estimand.instrumental_variables
        + estimate.estimator._effect_modifier_names,
    )

    if chosen_variables is None:
        logger.info("INFO: There are no chosen variables")
    else:
        logger.info("INFO: The chosen variables are: " + ",".join(chosen_variables))

    if probability_of_change is None and noise > 1:
        logger.error("Error in using noise:{} for Binary Flip. The value is greater than 1".format(noise))
        raise ValueError("The value for Binary Flip cannot be greater than 1")
    elif probability_of_change is None and noise <= 1:
        probability_of_change = noise
    elif probability_of_change > 1:
        logger.error(
            "The probability of flip is: {}, However, this value cannot be greater than 1".format(probability_of_change)
        )
        raise ValueError("Probability of Flip cannot be greater than 1")

    if sample_size > len(data):
        logger.warning("The sample size is larger than the population size")

    logger.info("Refutation over {} simulated datasets of size {} each".format(num_simulations, sample_size))

    sample_estimates = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_refute_once)(
            data, target_estimand, estimate, chosen_variables, random_state, sample_size, noise, probability_of_change
        )
        for _ in tqdm(
            range(num_simulations),
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            disable=not show_progress_bar,
            desc="Refuting Estimates: ",
        )
    )
    sample_estimates = np.array(sample_estimates)

    refute = CausalRefutation(
        estimate.value, np.mean(sample_estimates), refutation_type="Refute: Bootstrap Sample Dataset"
    )

    # We want to see if the estimate falls in the same distribution as the one generated by the refuter
    # Ideally that should be the case as running bootstrap should not have a significant effect on the ability
    # of the treatment to affect the outcome
    refute.add_significance_test_results(test_significance(estimate, sample_estimates))

    return refute
