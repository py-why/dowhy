from typing import Callable, List, Optional, Union

import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefutation
from dowhy.causal_refuters.add_unobserved_common_cause import sensitivity_simulation
from dowhy.causal_refuters.bootstrap_refuter import refute_bootstrap
from dowhy.causal_refuters.data_subset_refuter import refute_data_subset
from dowhy.causal_refuters.dummy_outcome_refuter import refute_dummy_outcome
from dowhy.causal_refuters.placebo_treatment_refuter import refute_placebo_treatment
from dowhy.causal_refuters.random_common_cause import refute_random_common_cause

ALL_REFUTERS = [
    sensitivity_simulation,
    refute_bootstrap,
    refute_data_subset,
    refute_dummy_outcome,
    refute_placebo_treatment,
    refute_random_common_cause,
    # TODO: Sensitivity Analyzers excluded from list due to different return type
]


def refute_estimate(
    data: pd.DataFrame,
    target_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    treatment_name: Optional[str] = None,
    outcome_name: Optional[str] = None,
    refuters: List[Callable[..., Union[CausalRefutation, List[CausalRefutation]]]] = ALL_REFUTERS,
    **kwargs,
) -> List[CausalRefutation]:
    """Executes a list of refuters using the default parameters
       Only refuters that return CausalRefutation or a list of CausalRefutation is supported

    :param data: pd.DataFrame: Data to run the refutation
    :param target_estimand: IdentifiedEstimand: Identified estimand to run the refutation
    :param estimate: CausalEstimate: Estimate to run the refutation
    :param treatment_name: str: Name of the treatment (Optional)
    :param outcome_name: str: Name of the outcome (Optional)
    :param refuters: list: List of refuters to execute
    :**kwargs: Replace any default for the provided list of refuters

    """
    refuter_kwargs = {
        "data": data,
        "target_estimand": target_estimand,
        "estimate": estimate,
        "treatment_name": treatment_name,
        "outcome_name": outcome_name,
        **kwargs,
    }

    results = []
    for refuter in refuters:
        refute = refuter(**refuter_kwargs)
        if isinstance(refute, list):
            results.extend(refute)
        else:
            results.append(refute)

    return results
