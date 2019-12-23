from os import path
from dowhy.causal_model import CausalModel


here = path.abspath(path.dirname(__file__))
# Loading version number
with open(path.join(here, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
