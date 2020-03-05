import pytest
import numpy
import random as rand

@pytest.fixture
def fixed_seed():
    rand.seed(0)
    numpy.random.seed(0)

