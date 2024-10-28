from importlib import import_module

from dowhy.graph_learner import GraphLearner


def get_discovery_class_object(method_name, *args, **kwargs):
    """
    Import class from graph_learners.

    """
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        module_name = method_name
        class_name = module_name.upper()

        discovery_module = import_module("." + module_name, package="dowhy.graph_learners")
        discovery_class = getattr(discovery_module, class_name)
        assert issubclass(discovery_class, GraphLearner)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError("{} is not an existing causal discovery method.".format(method_name))
    return discovery_class


def get_library_class_object(module_method_name, *args, **kwargs):
    """
    Import library for causal inference.

    """
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        (module_name, _, class_name) = module_method_name.rpartition(".")
        discovery_module = import_module(module_name)
        discovery_class = getattr(discovery_module, class_name)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError(
            "Error loading {}.{}. Double-check the method name and ensure that all library dependencies are installed.".format(
                module_name, class_name
            )
        )
    return discovery_class
