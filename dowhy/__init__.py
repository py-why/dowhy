import logging
from os import path
from dowhy.causal_model import CausalModel

logging.getLogger(__name__).addHandler(logging.NullHandler())

here = path.abspath(path.dirname(__file__))
# Loading version number
with open(path.join(here, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
