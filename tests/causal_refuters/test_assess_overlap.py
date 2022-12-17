import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_refuters.assess_overlap_overrule import SupportConfig


@pytest.fixture
def dummy_data():
    """
    Returns a dataframe with known violation of support and overlap.

    Two binary features (X1, X2) and a treatment indicator (T), where
    P(X1 = 1, X2 = 1) = 0
    P(T | X1 = 0, X2 = 0) = 0.5, and zero otherwise
    """

    return pd.DataFrame(
        np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
            * 50
        ),
        columns=["X1", "X2", "T", "Y"],
    )


@pytest.fixture
def refute(dummy_data):
    model = CausalModel(data=dummy_data, treatment="T", outcome="Y", common_causes=["X1", "X2"])
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    support_config = SupportConfig(seed=0)

    return model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="assess_overlap",
        support_config=support_config,
    )


class TestAssessOverlapRefuter(object):
    # TODO: Check directly for correct behavior, rather than checking the rules
    # themselves, which can be non-deterministic (all the following are equivalent)
    def test_rules_support(self, refute):
        """
        Check if the support rules cover the correct region.
        """
        assert refute.RS_support_estimator.rules() in [
            [[("X1", "not", "")], [("X1", "", ""), ("X2", "not", "")]],
            [[("X1", "not", "")], [("X2", "not", ""), ("X1", "", "")]],
            [[("X2", "not", "")], [("X1", "not", ""), ("X2", "", "")]],
            [[("X2", "not", "")], [("X2", "", ""), ("X1", "not", "")]],
            [[("X2", "not", "")], [("X1", "not", "")]],
            [[("X1", "not", "")], [("X2", "not", "")]],
        ]

    def test_rules_overlap(self, refute):
        assert refute.RS_overlap_estimator.rules() in [
            [[("X1", "not", ""), ("X2", "not", "")]],
            [[("X2", "not", ""), ("X1", "not", "")]],
        ]
