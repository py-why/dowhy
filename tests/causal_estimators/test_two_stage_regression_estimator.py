import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy import CausalModel
from dowhy.causal_estimators.two_stage_regression_estimator import TwoStageRegressionEstimator
from dowhy.causal_identifier import EstimandType

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestTwoStageRegressionEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "num_frontdoor_variables",
            "treatment_is_binary",
            "outcome_is_binary",
        ],
        [
            (
                0.1,
                TwoStageRegressionEstimator,
                [0],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    1,
                ],
                [False],
                [
                    False,
                ],
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
        num_frontdoor_variables,
        treatment_is_binary,
        outcome_is_binary,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method="frontdoor")
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            num_frontdoor_variables=num_frontdoor_variables,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                False,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )

    def test_frontdoor_estimator(self):
        """
        Test for frontdoor estimation, from @AlxndrMlk
        See issue #616 https://github.com/py-why/dowhy/issues/616
        """

        # Create the graph describing the causal structure
        graph = """
        graph [
            directed 1
            
            node [
                id "X" 
                label "X"
            ]    
            node [
                id "Z"
                label "Z"
            ]
            node [
                id "Y"
                label "Y"
            ]
            node [
                id "U"
                label "U"
            ]
            
            edge [
                source "X"
                target "Z"
            ]
            
            edge [
                source "Z"
                target "Y"
            ]
            
            edge [
                source "U"
                target "Y"
            ]
            
            edge [
                source "U"
                target "X"
            ]
        ]
        """.replace(
            "\n", ""
        )

        N_SAMPLES = 10000
        # Generate the data
        U = np.random.randn(N_SAMPLES)
        X = np.random.randn(N_SAMPLES) + 0.3 * U
        Z = 0.7 * X + 0.3 * np.random.randn(N_SAMPLES)
        Y = 0.65 * Z + 0.2 * U

        # Data to df
        df = pd.DataFrame(np.vstack([X, Z, Y]).T, columns=["X", "Z", "Y"])

        # Create a model
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=graph)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Estimate the effect with front-door
        estimate = model.estimate_effect(identified_estimand=estimand, method_name="frontdoor.two_stage_regression")
        assert estimate.value == pytest.approx(0.45, 0.025)

    @mark.parametrize(
        [
            "Estimator",
            "num_treatments",
            "num_frontdoor_variables",
        ],
        [
            (
                TwoStageRegressionEstimator,
                [2, 1],
                [1, 2],
            )
        ],
    )
    def test_frontdoor_num_variables_error(self, Estimator, num_treatments, num_frontdoor_variables):
        estimator_tester = SimpleEstimator(error_tolerance=0, Estimator=Estimator, identifier_method="frontdoor")
        with pytest.raises((ValueError, Exception)):
            estimator_tester.average_treatment_effect_testsuite(
                num_common_causes=[1, 1],
                num_instruments=[0, 0],
                num_effect_modifiers=[0, 0],
                num_treatments=num_treatments,
                num_frontdoor_variables=num_frontdoor_variables,
                treatment_is_binary=[True],
                outcome_is_binary=[False],
                confidence_intervals=[
                    True,
                ],
                test_significance=[
                    False,
                ],
                method_params={"num_simulations": 10, "num_null_simulations": 10},
            )


def _make_mediation_data(n=2000, seed=42):
    """Generate linear mediation data with known NIE and NDE.

    Graph:  X -> M -> Y,  X -> Y  (no confounders)
    True coefficients:
        X -> M: 0.5  (alpha)
        M -> Y: 0.8  (beta)
        X -> Y: 0.3  (gamma, direct)
    True NIE = alpha * beta = 0.4
    True NDE = gamma = 0.3
    True ATE = NIE + NDE = 0.7
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


_MEDIATION_GML = """
graph [
    directed 1
    node [ id "X" label "X" ]
    node [ id "M" label "M" ]
    node [ id "Y" label "Y" ]
    edge [ source "X" target "M" ]
    edge [ source "X" target "Y" ]
    edge [ source "M" target "Y" ]
]
""".replace(
    "\n", " "
)


class TestTwoStageRegressionMediationNIE:
    """Tests for NIE (Natural Indirect Effect) estimation via mediation."""

    def test_nie_estimate_close_to_true_value(self):
        """NIE estimate should be close to the true alpha*beta = 0.4."""
        df = _make_mediation_data()
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand = model.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            proceed_when_unidentifiable=True,
        )
        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="mediation.two_stage_regression",
        )
        assert estimate.value == pytest.approx(0.4, abs=0.1)

    def test_nie_estimate_is_scalar(self):
        """The NIE estimate value must be a scalar."""
        df = _make_mediation_data()
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand = model.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            proceed_when_unidentifiable=True,
        )
        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="mediation.two_stage_regression",
        )
        assert np.isscalar(estimate.value) or (isinstance(estimate.value, np.ndarray) and estimate.value.ndim == 0)


class TestTwoStageRegressionMediationNDE:
    """Tests for NDE (Natural Direct Effect) estimation via mediation.

    Regression test for the bug where ``modified_target_estimand`` was
    re-assigned to a fresh ``copy.deepcopy`` *after* its ``identifier_method``
    had been set to ``"backdoor"``, which caused the NDE second-stage model to
    receive the wrong estimand and fail (or silently produce wrong results).
    """

    def test_nde_estimate_close_to_true_value(self):
        """NDE estimate should be close to the true direct effect gamma = 0.3."""
        df = _make_mediation_data()
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand = model.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NDE,
            proceed_when_unidentifiable=True,
        )
        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="mediation.two_stage_regression",
        )
        assert estimate.value == pytest.approx(0.3, abs=0.1)

    def test_nde_plus_nie_equals_ate(self):
        """NDE + NIE should equal the total ATE (= 0.7)."""
        df = _make_mediation_data()

        model_nie = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand_nie = model_nie.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            proceed_when_unidentifiable=True,
        )
        nie = model_nie.estimate_effect(
            identified_estimand=estimand_nie,
            method_name="mediation.two_stage_regression",
        ).value

        model_nde = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand_nde = model_nde.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NDE,
            proceed_when_unidentifiable=True,
        )
        nde = model_nde.estimate_effect(
            identified_estimand=estimand_nde,
            method_name="mediation.two_stage_regression",
        ).value

        # ATE via simple regression for reference
        from sklearn.linear_model import LinearRegression

        ate_lr = LinearRegression().fit(df[["X"]], df["Y"]).coef_[0]
        assert nde + nie == pytest.approx(ate_lr, abs=0.1)

    def test_nde_estimand_uses_correct_backdoor_variables(self):
        """_second_stage_model_nde must use mediation_second_stage_confounders.

        After the bug fix, identifier_method must be 'backdoor' and
        backdoor_variables must equal mediation_second_stage_confounders on the
        NDE second-stage model, not the raw target_estimand backdoor_variables.
        """
        df = _make_mediation_data()
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=_MEDIATION_GML)
        estimand = model.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NDE,
            proceed_when_unidentifiable=True,
        )
        estimator = TwoStageRegressionEstimator(identified_estimand=estimand)
        nde_estimand = estimator._second_stage_model_nde._target_estimand
        assert nde_estimand.identifier_method == "backdoor"
        assert nde_estimand.backdoor_variables == estimand.mediation_second_stage_confounders
