import re
import string
from importlib import import_module

from dowhy.interpreter import Interpreter


def _camel_to_snake(name):
    """Convert CamelCase names to snake_case module names."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_class_object(method_name, *args, **kwargs):
    # from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
    try:
        if "_" in method_name:
            module_name = method_name
            class_name = string.capwords(method_name, "_").replace("_", "")
        else:
            module_name = _camel_to_snake(method_name)
            class_name = method_name

        interpreter_module = import_module("." + module_name, package="dowhy.interpreters")
        interpreter_class = getattr(interpreter_module, class_name)
        assert issubclass(interpreter_class, Interpreter)

    except (AttributeError, AssertionError, ImportError):
        raise ImportError("{} is not an existing interpreter.".format(method_name))
    return interpreter_class
