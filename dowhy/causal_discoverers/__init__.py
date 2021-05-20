import string
from importlib import import_module

from dowhy.causal_discovery import CausalDiscovery

def get_class_object(method_name, *args, **kwargs):
	# from https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version
	try:
		module_name = method_name
		class_name = module_name.upper()

		discovery_module = import_module('.' + module_name, package="dowhy.causal_discoverers")
		discovery_class = getattr(discovery_module, class_name)
		assert issubclass(discovery_class, CausalDiscovery)

	except (AttributeError, AssertionError, ImportError):
		raise ImportError('{} is not an existing causal discovery method.'.format(method_name))
	return discovery_class
