import random as rand

import numpy
import pytest
import torch


@pytest.fixture
def fixed_seed():
    rand.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(0)
