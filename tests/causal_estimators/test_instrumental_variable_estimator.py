import itertools

import numpy as np
import pandas as pd
import pytest
from pytest import mark

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

    def test_iv_with_nonzero_mean_instruments(self):
        """
        Regression test for issue #1821: IV estimator with 2+ instruments
        should fit 2SLS with an intercept to handle non-mean-zero instruments.
        This test uses binary instruments coded as {0, 1}, which have non-zero mean.
        The true treatment effect should be recoverable with proper constant term.
        """
        rng = np.random.default_rng(42)
        n = 5000
        # Binary instruments with non-zero mean (coded as 0/1)
        z1 = rng.binomial(1, 0.5, n).astype(float)
        z2 = rng.binomial(1, 0.5, n).astype(float)
        # Confounding (unobserved)
        u = rng.normal(size=n)
        # Treatment with non-zero mean
        x = 3.0 + z1 + z2 + u + rng.normal(size=n)
        # Outcome: true effect is 2.0
        y = 10.0 + 2.0 * x + 2 * u + rng.normal(size=n)

        df = pd.DataFrame(dict(Z1=z1, Z2=z2, X=x, Y=y))
        # DAG: Z1, Z2 -> X -> Y, U -> X, U -> Y
        g = "digraph{Z1->X;Z2->X;U->X;U->Y;X->Y}"

        model = CausalModel(df, "X", "Y", graph=g)
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate with both instruments
        estimate = model.estimate_effect(
            identified_estimand, method_name="iv.instrumental_variable"
        )

        # The estimate should be close to the true effect of 2.0
        # With the bug (no intercept), this would be ~4.2
        # With the fix (intercept included), this should be ~2.0
        assert (
            abs(estimate.value - 2.0) < 0.5
        ), f"IV estimate {estimate.value} is too far from true effect 2.0"

