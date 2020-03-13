import pytest
import numpy as np
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    @pytest.mark.parameterize(["error_tolerance","estimator_method"],
                              [0.01, "iv.instrument_variable"])
    def test_refutation_placebo_refuter_continueous(self, error_tolerance, estimator_method):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "data_subset_refuter")
        refuter_tester.continuous_treatment_testsuite() # Run both
        