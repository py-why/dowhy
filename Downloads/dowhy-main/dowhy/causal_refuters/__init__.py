import string
from importlib import import_module

from dowhy.causal_refuter import CausalRefuter
from dowhy.causal_refuters.add_unobserved_common_cause import (
    AddUnobservedCommonCause,
    sensitivity_e_value,
    sensitivity_simulation,
)
from dowhy.causal_refuters.bootstrap_refuter import BootstrapRefuter, refute_bootstrap
from dowhy.causal_refuters.data_subset_refuter import DataSubsetRefuter, refute_data_subset
from dowhy.causal_refuters.dummy_outcome_refuter import DummyOutcomeRefuter, refute_dummy_outcome
from dowhy.causal_refuters.placebo_treatment_refuter import PlaceboTreatmentRefuter, refute_placebo_treatment
from dowhy.causal_refuters.random_common_cause import RandomCommonCause, refute_random_common_cause
from dowhy.causal_refuters.refute_estimate import refute_estimate


def get_class_object(method_name, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = string.capwords(method_name, "_").replace("_", "")

        refuter_module = import_module("." + module_name, package="dowhy.causal_refuters")
        refuter_class = getattr(refuter_module, class_name)
        assert issubclass(refuter_class, CausalRefuter)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError("{} is not an existing causal refuter.".format(method_name))
    return refuter_class


__all__ = [
    "AddUnobservedCommonCause",
    "BootstrapRefuter",
    "DataSubsetRefuter",
    "PlaceboTreatmentRefuter",
    "RandomCommonCause",
    "DummyOutcomeRefuter",
    "refute_bootstrap",
    "refute_data_subset",
    "refute_random_common_cause",
    "refute_placebo_treatment",
    "sensitivity_simulation",
    "sensitivity_e_value",
    "refute_dummy_outcome",
    "refute_estimate",
]
