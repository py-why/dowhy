import string
from importlib import import_module

from dowhy.causal_estimator import CausalEstimator

def get_class_object(method_name, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = string.capwords(method_name, "_").replace('_', '')

        estimator_module = import_module('.' + module_name, package="dowhy.causal_estimators")
        estimator_class = getattr(estimator_module, class_name)
        assert issubclass(estimator_class, CausalEstimator)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError('{} is not an existing causal estimator.'.format(method_name))
    return estimator_class
