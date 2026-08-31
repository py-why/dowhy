import itertools

import numpy as np
import pytest
from pytest import mark

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestInstrumentalVariableEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.4,
                InstrumentalVariableEstimator,
                [0, 1],
                [1, 2],
                [
                    0,
                ],
                [1, 2],
                [False, True],
                [
                    False,
                ],
                "iv",
            ),
        ],
    )
    def test_average_treatment_effect(
        self,
        error_tolerance,
        Estimator,
        num_common_causes,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        treatment_is_binary,
        outcome_is_binary,
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        # Not using testsuite from .base/TestEstimtor, custom code below
        args_dict = {
            "num_common_causes": num_common_causes,
            "num_instruments": num_instruments,
            "num_effect_modifiers": num_effect_modifiers,
            "num_treatments": num_treatments,
            "treatment_is_binary": treatment_is_binary,
            "outcome_is_binary": outcome_is_binary,
        }
        keys, values = zip(*args_dict.items())
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for cfg in configs:
            print("\nConfig:", cfg)
            cfg["method_params"] = {}
            if cfg["num_instruments"] >= cfg["num_treatments"]:
                estimator_tester.average_treatment_effect_test(**cfg)
            else:
                with pytest.raises(ValueError):
                    estimator_tester.average_treatment_effect_test(**cfg)

        # More cases where Exception  is expected
        cfg = configs[0]
        cfg["num_instruments"] = 0
        with pytest.raises(ValueError):
            estimator_tester.average_treatment_effect_test(**cfg)


def test_iv_estimator_raises_on_constant_instrument():
    """Instrument with a single unique value must raise ValueError, not silently return NaN."""
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=1,
        num_instruments=1,
        num_treatments=1,
        num_samples=500,
        treatment_is_binary=True,
    )
    inst = data["instrument_names"][0]
    df = data["df"].copy()
    # Replace the instrument with a constant value so it has no variation.
    df[inst] = 0.0

    model = CausalModel(
        data=df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
        proceed_when_unidentifiable=True,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    with pytest.raises(ValueError, match="at least 2 distinct values"):
        model.estimate_effect(estimand, method_name="iv.instrumental_variable")
