import logging
from dowhy.causal_model import CausalModel
from . import _version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = _version.get_versions()["version"]
