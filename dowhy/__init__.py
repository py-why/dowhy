import logging
from os import path
from dowhy.causal_model import CausalModel
logging.getLogger(__name__).addHandler(logging.NullHandler())

here = path.abspath(path.dirname(__file__))


# 0.0.0 is standard placeholder for poetry-dynamic-versioning
__version__ = "0.0.0"
