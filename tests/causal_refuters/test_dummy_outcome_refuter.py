import pytest
import numpy as np
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestDummyOtcomeRefuter(object):
    @pytest.mark.parametrize(["error_tolerence","estimator_method"],
                             [(0.03, "iv.instrumental_variable")])
    def test_refutation_dummy_outcome_refuter(self, error_tolerence, estimator_method):
        refuter_tester = TestRefuter(error_tolerence, estimator_method, "dummy_outcome_refuter")
        refuter_tester.continuous_treatment_testsuite()