import logging
from dowhy.causal_model import CausalModel
from . import _version

logging.getLogger(__name__).addHandler(logging.NullHandler())

# 0.0.0 is standard placeholder for poetry-dynamic-versioning
__version__ = "0.0.0"