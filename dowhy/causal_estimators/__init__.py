import string
from importlib import import_module

from dowhy.causal_estimator import CausalEstimator


def get_class_object(method_name, estimator_name=None, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = string.capwords(method_name, "_").replace("_", "")

        estimator_module = import_module("." + module_name, package="dowhy.causal_estimators")
        estimator_class = getattr(estimator_module, class_name)
        assert issubclass(estimator_class, CausalEstimator)

    except (AttributeError, AssertionError, ImportError):
        # Handle externally provided estimator classes
        try:
            module_name = ".".join(estimator_name.split(".")[:-1])
            classname = estimator_name.split(".")[-1]
            estimator_class = getattr(import_module(module_name), classname)
            assert issubclass(estimator_class, CausalEstimator)

        except (AttributeError, AssertionError, ImportError):
            est_name = method_name if estimator_name is None else estimator_name
            raise ImportError("{} is not an existing causal estimator.".format(est_name))

    return estimator_class
