import random as rand

import numpy
import pytest


@pytest.fixture
def fixed_seed():
    rand.seed(0)
    numpy.random.seed(0)
