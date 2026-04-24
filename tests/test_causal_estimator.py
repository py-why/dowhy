import unittest

import pytest

import dowhy
import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimator


class MockEstimator(CausalEstimator):
    pass


def test_causal_estimator_placeholder_methods():
    estimator = MockEstimator(None)
    with pytest.raises(NotImplementedError):
        estimator._do(None)
    with pytest.raises(NotImplementedError):
        estimator.construct_symbolic_estimator(None)


def test_nan_warning_emitted_when_outcome_has_nan(caplog):
    """estimate_effect warns when the outcome column contains NaN values (closes #827)."""
    import logging

    data = dowhy.datasets.linear_dataset(
        beta=10, num_common_causes=2, num_samples=200, num_instruments=0, num_effect_modifiers=0
    )
    df = data["df"].copy()
    df.loc[0, data["outcome_name"]] = float("nan")

    model = CausalModel(
        data=df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    with caplog.at_level(logging.WARNING, logger="dowhy.causal_estimator"):
        model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    assert any(
        "NaN" in record.message for record in caplog.records
    ), "Expected a NaN-warning log record, but none was found."


def test_no_nan_warning_when_data_is_clean(caplog):
    """estimate_effect does not warn when the data contains no NaN values."""
    import logging

    data = dowhy.datasets.linear_dataset(
        beta=10, num_common_causes=2, num_samples=200, num_instruments=0, num_effect_modifiers=0
    )

    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    with caplog.at_level(logging.WARNING, logger="dowhy.causal_estimator"):
        model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    nan_warnings = [r for r in caplog.records if "NaN" in r.message]
    assert len(nan_warnings) == 0, f"Unexpected NaN warning(s): {[r.message for r in nan_warnings]}"


class TestCausalEstimator(unittest.TestCase):
    def setUp(self):
        self.df = None
        self.ate = 1


if __name__ == "__main__":
    unittest.main()
