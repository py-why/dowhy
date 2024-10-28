import string
from importlib import import_module

from dowhy.do_sampler import DoSampler

PACKAGE_NAME = "dowhy.do_samplers"


def get_class_object(method_name, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = string.capwords(method_name, "_").replace("_", "")
        do_sampler_module = import_module("." + module_name, package=PACKAGE_NAME)
        do_sampler_class = getattr(do_sampler_module, class_name)
        assert issubclass(do_sampler_class, DoSampler)

    except (AttributeError, AssertionError, ImportError) as e:
        if isinstance(e, ImportError) and e.name != PACKAGE_NAME + "." + module_name:
            raise e
        raise ImportError("{} is not an existing do sampler.".format(method_name))
    return do_sampler_class
