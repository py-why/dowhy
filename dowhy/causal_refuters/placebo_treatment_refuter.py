import copy
import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
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


def _get_placebo_names(treatment_names: List[str]) -> List[str]:
    """Return placebo column name(s) for the given treatment name(s).

    Single-treatment case uses ``"placebo"`` for backward compatibility;
    multi-treatment case prefixes each name with ``"placebo_"``.
    """
    if len(treatment_names) == 1:
        return ["placebo"]
    return ["placebo_" + t for t in treatment_names]


def _generate_random_placebo(data: pd.DataFrame, treatment_name: str, type_dict: Dict) -> pd.Series:
    """Generate a single random placebo column matching the dtype of *treatment_name*."""
    dtype = type_dict[treatment_name]
    n = data.shape[0]
    if pd.api.types.is_float_dtype(dtype):
        logger.info(
            "Using a Normal Distribution with Mean:{} and Variance:{}".format(
                DEFAULT_MEAN_OF_NORMAL,
                DEFAULT_STD_DEV_OF_NORMAL,
            )
        )
        return pd.Series(
            np.random.randn(n) * DEFAULT_STD_DEV_OF_NORMAL + DEFAULT_MEAN_OF_NORMAL,
            index=data.index,
        )
    elif pd.api.types.is_bool_dtype(dtype):
        logger.info(
            "Using a Binomial Distribution with {} trials and {} probability of success".format(
                DEFAULT_NUMBER_OF_TRIALS,
                DEFAULT_PROBABILITY_OF_BINOMIAL,
            )
        )
        return pd.Series(
            np.random.binomial(DEFAULT_NUMBER_OF_TRIALS, DEFAULT_PROBABILITY_OF_BINOMIAL, n).astype(bool),
            index=data.index,
        )
    elif pd.api.types.is_integer_dtype(dtype):
        logger.info(
            "Using a Discrete Uniform Distribution lying between {} and {}".format(
                data[treatment_name].min(), data[treatment_name].max()
            )
        )
        return pd.Series(
            np.random.randint(low=data[treatment_name].min(), high=data[treatment_name].max() + 1, size=n),
            index=data.index,
        )
    elif isinstance(dtype, pd.CategoricalDtype):
        treatment = data[treatment_name]
        categories = treatment.cat.categories
        logger.info("Using a Discrete Uniform Distribution with the following categories:{}".format(categories))
        return pd.Series(
            pd.Categorical(
                np.random.choice(categories, size=n),
                categories=categories,
                ordered=treatment.cat.ordered,
            ),
            index=data.index,
        )
    raise ValueError("Unsupported treatment dtype '{}' for treatment '{}'.".format(dtype, treatment_name))


def _refute_once(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    treatment_names: List[str],
    type_dict: Dict,
    placebo_type: PlaceboType = PlaceboType.DEFAULT,
    random_state: Optional[np.random.RandomState] = None,
):
    placebo_names = _get_placebo_names(treatment_names)

    if placebo_type == PlaceboType.PERMUTE:
        if random_state is None:
            permuted_idx = np.random.choice(data.shape[0], size=data.shape[0], replace=False)
        else:
            permuted_idx = random_state.choice(data.shape[0], size=data.shape[0], replace=False)
        new_data = data.copy()
        for t, pname in zip(treatment_names, placebo_names):
            permuted_col = data[t].iloc[permuted_idx]
            # Reset index so assignment is by position, not label alignment,
            # while preserving the column's original dtype (e.g. CategoricalDtype).
            new_data[pname] = permuted_col.set_axis(data.index)
        if target_estimand.identifier_method.startswith("iv"):
            new_instruments_values = data[estimate.estimator.estimating_instrument_names].iloc[permuted_idx].values
            new_instruments_df = pd.DataFrame(
                new_instruments_values,
                columns=["placebo_" + s for s in data[estimate.estimator.estimating_instrument_names].columns],
            )
    else:
        new_data = data.copy()
        for t, pname in zip(treatment_names, placebo_names):
            new_data[pname] = _generate_random_placebo(data, t, type_dict)

    if target_estimand.identifier_method.startswith("iv"):
        new_data = pd.concat((new_data, new_instruments_df), axis=1)
    # Sanity check the data
    logger.debug(new_data[0:10])
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

    # For IV methods, the estimating_instrument_names should also be
    # changed. Create a copy to avoid modifying original object.
    # estimate is a CausalEstimate; the actual estimator is estimate.estimator.
    if hasattr(estimate, "estimator") and isinstance(estimate.estimator, InstrumentalVariableEstimator):
        estimate = copy.deepcopy(estimate)
        if estimate.estimator.iv_instrument_name is not None:
            estimate.estimator.iv_instrument_name = [
                "placebo_" + s for s in parse_state(estimate.estimator.iv_instrument_name)
            ]

    # We need to change the identified estimand
    # We make a copy as a safety measure, we don't want to change the
    # original DataFrame
    identified_estimand = copy.deepcopy(target_estimand)
    identified_estimand.treatment_variable = _get_placebo_names(treatment_names)

    if target_estimand.identifier_method.startswith("iv"):
        identified_estimand.instrumental_variables = [
            "placebo_" + s for s in identified_estimand.instrumental_variables
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
        data=data,
        treatment_name=estimate._treatment_name,
        outcome_name=estimate._outcome_name,
        estimate=0,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        target_estimand=estimate.target_estimand,
        realized_estimand_expr=estimate.realized_estimand_expr,
    )

    refute.add_significance_test_results(test_significance(dummy_estimator, sample_estimates))

    return refute
