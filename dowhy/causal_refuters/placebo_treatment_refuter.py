import copy
import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from dowhy.causal_estimator import CausalEstimate, CausalEstimator
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefutation, CausalRefuter, test_significance
from dowhy.utils.api import parse_state

logger = logging.getLogger(__name__)


# Default value of the p value taken for the distribution
DEFAULT_PROBABILITY_OF_BINOMIAL = 0.5
# Number of Trials: Number of cointosses to understand if a sample gets the treatment
DEFAULT_NUMBER_OF_TRIALS = 1
# Mean of the Normal Distribution
DEFAULT_MEAN_OF_NORMAL = 0
# Standard Deviation of the Normal Distribution
DEFAULT_STD_DEV_OF_NORMAL = 0


class PlaceboType(Enum):
    DEFAULT = "Random Data"
    PERMUTE = "permute"


class PlaceboTreatmentRefuter(CausalRefuter):
    """Refute an estimate by replacing treatment with a randomly-generated placebo variable.

    Supports additional parameters that can be specified in the refute_estimate() method. For joblib-related parameters (n_jobs, verbose), please refer to the joblib documentation for more details (https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html).

    :param placebo_type: Default is to generate random values for the treatment. If placebo_type is "permute", then the original treatment values are permuted by row.
    :type placebo_type: str, optional

    :param num_simulations: The number of simulations to be run, which is ``CausalRefuter.DEFAULT_NUM_SIMULATIONS`` by default
    :type num_simulations: int, optional

    :param random_state: The seed value to be added if we wish to repeat the same random behavior. If we want to repeat the same behavior we push the same seed in the psuedo-random generator.
    :type random_state: int, RandomState, optional

    :param n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all (this is the default).
    :type n_jobs: int, optional

    :param verbose: The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported. The default is 0.
    :type verbose: int, optional
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._placebo_type = kwargs.pop("placebo_type", None)
        if self._placebo_type is None:
            self._placebo_type = "Random Data"
        self._num_simulations = kwargs.pop("num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS)
        self._random_state = kwargs.pop("random_state", None)

        self.logger = logging.getLogger(__name__)

    def refute_estimate(self, show_progress_bar=False):
        refute = refute_placebo_treatment(
            data=self._data,
            target_estimand=self._target_estimand,
            estimate=self._estimate,
            treatment_names=self._treatment_name,
            num_simulations=self._num_simulations,
            placebo_type=PlaceboType(self._placebo_type),
            random_state=self._random_state,
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
    treatment_names: List[str],
    type_dict: Dict,
    placebo_type: PlaceboType = PlaceboType.DEFAULT,
    random_state: Optional[np.random.RandomState] = None,
):
    if placebo_type == PlaceboType.PERMUTE:
        permuted_idx = None
        if random_state is None:
            permuted_idx = np.random.choice(data.shape[0], size=data.shape[0], replace=False)

        else:
            permuted_idx = random_state.choice(data.shape[0], size=data.shape[0], replace=False)
        new_treatment = data[treatment_names].iloc[permuted_idx].values
        if target_estimand.identifier_method.startswith("iv"):
            new_instruments_values = data[estimate.estimator.estimating_instrument_names].iloc[permuted_idx].values
            new_instruments_df = pd.DataFrame(
                new_instruments_values,
                columns=["placebo_" + s for s in data[estimate.estimator.estimating_instrument_names].columns],
            )
    else:
        if "float" in type_dict[treatment_names[0]].name:
            logger.info(
                "Using a Normal Distribution with Mean:{} and Variance:{}".format(
                    DEFAULT_MEAN_OF_NORMAL,
                    DEFAULT_STD_DEV_OF_NORMAL,
                )
            )
            new_treatment = np.random.randn(data.shape[0]) * DEFAULT_STD_DEV_OF_NORMAL + DEFAULT_MEAN_OF_NORMAL

        elif "bool" in type_dict[treatment_names[0]].name:
            logger.info(
                "Using a Binomial Distribution with {} trials and {} probability of success".format(
                    DEFAULT_NUMBER_OF_TRIALS,
                    DEFAULT_PROBABILITY_OF_BINOMIAL,
                )
            )
            new_treatment = np.random.binomial(
                DEFAULT_NUMBER_OF_TRIALS,
                DEFAULT_PROBABILITY_OF_BINOMIAL,
                data.shape[0],
            ).astype(bool)

        elif "int" in type_dict[treatment_names[0]].name:
            logger.info(
                "Using a Discrete Uniform Distribution lying between {} and {}".format(
                    data[treatment_names[0]].min(), data[treatment_names[0]].max()
                )
            )
            new_treatment = np.random.randint(
                low=data[treatment_names[0]].min(), high=data[treatment_names[0]].max() + 1, size=data.shape[0]
            )

        elif "category" in type_dict[treatment_names[0]].name:
            categories = data[treatment_names[0]].unique()
            logger.info("Using a Discrete Uniform Distribution with the following categories:{}".format(categories))
            sample = np.random.choice(categories, size=data.shape[0])
            new_treatment = pd.Series(sample, index=data.index).astype("category")

    # Create a new column in the data by the name of placebo
    new_data = data.assign(placebo=new_treatment)
    if target_estimand.identifier_method.startswith("iv"):
        new_data = pd.concat((new_data, new_instruments_df), axis=1)
    # Sanity check the data
    logger.debug(new_data[0:10])
    new_estimator = CausalEstimator.get_estimator_object(new_data, target_estimand, estimate)
    new_effect = new_estimator.estimate_effect()
    return new_effect.value


def refute_placebo_treatment(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    treatment_names: List,
    num_simulations: int = 100,
    placebo_type: PlaceboType = PlaceboType.DEFAULT,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    show_progress_bar: bool = False,
    n_jobs: int = 1,
    verbose: int = 0,
    **_,
) -> CausalRefutation:
    """Refute an estimate by replacing treatment with a randomly-generated placebo variable.

    :param data: pd.DataFrame: Data to run the refutation
    :param target_estimand: IdentifiedEstimand: Identified estimand to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param treatment_names: list: List of treatments
    :param num_simulations: The number of simulations to be run, which defaults to ``CausalRefuter.DEFAULT_NUM_SIMULATIONS``
    :param placebo_type: Default is to generate random values for the treatment. If placebo_type is "permute", then the original treatment values are permuted by row.
    :param random_state: The seed value to be added if we wish to repeat the same random behavior. If we want to repeat the same behavior we push the same seed in the psuedo-random generator.
    :param n_jobs: The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all (this is the default).
    :param verbose: The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported. The default is 0.
    """

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # only permute is supported for iv methods
    if target_estimand.identifier_method.startswith("iv"):
        if placebo_type != PlaceboType.PERMUTE:
            logger.error(
                "Only placebo_type=''permute'' is supported for creating placebo for instrumental variable estimation methods"
            )
            raise ValueError(
                "Only placebo_type=''permute'' is supported for creating placebo for instrumental variable estimation methods."
            )

    # We need to change the identified estimand
    # We make a copy as a safety measure, we don't want to change the
    # original DataFrame
    identified_estimand = copy.deepcopy(target_estimand)
    identified_estimand.treatment_variable = ["placebo"]

    if target_estimand.identifier_method.startswith("iv"):
        identified_estimand.instrumental_variables = [
            "placebo_" + s for s in identified_estimand.instrumental_variables
        ]
        # For IV methods, the estimating_instrument_names should also be
        # changed. Create a copy to avoid modifying original object
        if estimate.params["method_params"] is not None and "iv_instrument_name" in estimate.params["method_params"]:
            estimate = copy.deepcopy(estimate)
            estimate.params["method_params"]["iv_instrument_name"] = [
                "placebo_" + s for s in parse_state(estimate.params["method_params"]["iv_instrument_name"])
            ]

    logger.info("Refutation over {} simulated datasets of {} treatment".format(num_simulations, placebo_type))

    type_dict = dict(data.dtypes)

    # Run refutation in parallel
    sample_estimates = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_refute_once)(
            data, identified_estimand, estimate, treatment_names, type_dict, placebo_type, random_state
        )
        for _ in tqdm(
            range(num_simulations),
            disable=not show_progress_bar,
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            desc="Refuting Estimates: ",
        )
    )

    sample_estimates = np.array(sample_estimates)

    refute = CausalRefutation(
        estimate.value, np.mean(sample_estimates), refutation_type="Refute: Use a Placebo Treatment"
    )

    # Note: We hardcode the estimate value to ZERO as we want to check if it falls in the distribution of the refuter
    # Ideally we should expect that ZERO should fall in the distribution of the effect estimates as we have severed any causal
    # relationship between the treatment and the outcome.
    dummy_estimator = CausalEstimate(
        estimate=0,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        target_estimand=estimate.target_estimand,
        realized_estimand_expr=estimate.realized_estimand_expr,
    )

    refute.add_significance_test_results(test_significance(dummy_estimator, sample_estimates))

    return refute
