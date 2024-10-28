"""This module provides functionality to estimate confidence intervals via bootstrapping the fitting and sampling."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from dowhy.gcm import auto
from dowhy.gcm.causal_models import InvertibleStructuralCausalModel, ProbabilisticCausalModel, StructuralCausalModel
from dowhy.gcm.fitting_sampling import fit

# A convenience function when computing confidence intervals specifically for non-deterministic causal queries. This
# function evaluates the provided causal query multiple times to build a confidence interval based on the returned
# results.
# Note that this function does not re-fit the causal model(s) and only executes the provided query as it is. In order
# to re-refit the graphical causal model on random subsets of the data before executing the query, consider using the
# fit_and_compute function.
#
# **Example usage:**
#
#     >>> gcm.fit(causal_model, data)
#     >>> strength_medians, strength_intervals = gcm.confidence_intervals(
#     >>>     gcm.bootstrap_sampling(gcm.arrow_strength, causal_model, target_node='Y'))
#
# In this example, gcm.confidence_intervals is expecting a callable with non-deterministic outputs for building the
# confidence intervals. Since each causal query potentially expects a different set of parameters, we use 'partial'
# here to configure the function call. In this case,
# gcm.bootstrap_sampling(gcm.arrow_strength, causal_model, target_node='Y') would be equivalent to
# lambda : gcm.arrow_strength(causal_model, target_node='Y').
#
# In order to incorporate uncertainties coming from fitting the causal model(s), we can use
# gcm.fit_and_compute instead:
# >>>  strength_medians, strength_intervals = gcm.confidence_intervals(
# >>>        gcm.fit_and_compute(gcm.arrow_strength,
# >>>                                            causal_model,
# >>>                                            bootstrap_training_data=data,
# >>>                                            target_node='Y'))
# This would refit the provided causal_model on a subset of the data first before executing gcm.arrow_strength in each
# run.
bootstrap_sampling = partial


def fit_and_compute(
    f: Callable[
        [Union[ProbabilisticCausalModel, StructuralCausalModel, InvertibleStructuralCausalModel], Any],
        Dict[Any, Union[np.ndarray, float]],
    ],
    causal_model: Union[ProbabilisticCausalModel, StructuralCausalModel, InvertibleStructuralCausalModel],
    bootstrap_training_data: pd.DataFrame,
    bootstrap_data_subset_size_fraction: float = 0.75,
    auto_assign_quality: Optional[auto.AssignmentQuality] = None,
    *args,
    **kwargs,
):
    """A convenience function when computing confidence intervals specifically for causal queries. This function
    specifically bootstraps training *and* sampling.

    **Example usage:**

        >>> scores_median, scores_intervals = gcm.confidence_intervals(
        >>>     gcm.fit_and_compute(gcm.arrow_strength,
        >>>                         causal_model,
        >>>                         bootstrap_training_data=data,
        >>>                         target_node='Y'))

    :param f: The causal query to perform. A causal query is a function taking a graphical causal model as first
              parameter and an arbitrary number of remaining parameters. It must return a dictionary with
              attribution-like data.
    :param causal_model: A graphical causal model to perform the causal query on. It need not be fitted.
    :param bootstrap_training_data: The training data to use when fitting. A random subset from this data set is used
                                    in every iteration when calling fit.
    :param bootstrap_data_subset_size_fraction: The fraction defines the fractional size of the subset compared to
                                                the total training data.
    :param auto_assign_quality: If a quality is provided, then the existing causal mechanisms in the given causal_model
                                are overridden by new automatically inferred mechanisms based on the provided
                                AssignmentQuality. If None is given, the existing assigned mechanisms are used.
    :param args: Args passed through verbatim to the causal queries.
    :param kwargs: Keyword args passed through verbatim to the causal queries.
    :return: A tuple containing (1) the median of causal query results and (2) the confidence intervals.
    """

    def snapshot():
        causal_model_copy = causal_model.clone()
        sampled_data = bootstrap_training_data.iloc[
            np.random.choice(
                bootstrap_training_data.shape[0],
                int(bootstrap_training_data.shape[0] * bootstrap_data_subset_size_fraction),
                replace=False,
            )
        ]

        if auto_assign_quality is not None:
            auto.assign_causal_mechanisms(causal_model_copy, sampled_data, auto_assign_quality, override_models=True)

        fit(causal_model_copy, sampled_data)
        return f(causal_model_copy, *args, **kwargs)

    return snapshot
