import string
from importlib import import_module

from dowhy.causal_refuter import CausalRefuter


def get_class_object(method_name, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = string.capwords(method_name, "_").replace("_", "")

        refuter_module = import_module('.' + module_name,
                                       package="dowhy.causal_refuters")
        refuter_class = getattr(refuter_module, class_name)
        assert issubclass(refuter_class, CausalRefuter)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError('{} is not an existing causal refuter.'.format(method_name))
    return refuter_class
