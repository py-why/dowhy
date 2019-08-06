from os import path

here = path.abspath(path.dirname(__file__))
# Loading version number
with open(path.join(here, path.pardir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
