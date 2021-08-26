import pytest
import itertools

from dowhy import CausalModel
from dowhy import datasets

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures

econml = pytest.importorskip("econml")


class TestEconMLEstimator:
    """Smoke tests for the integration with EconML estimators

    These tests only check that the ate estimation routine can be executed without errors.
    We don't check the accuracy of the ate estimates as we don't want to take dependencies on
    EconML estimators.
    """

    def test_backdoor_estimators(self):
        # Setup data
        data = datasets.linear_dataset(
            10, num_common_causes=4, num_samples=10000,
            num_instruments=2, num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=False)
        df = data['df']
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"]
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True)
        # Test LinearDML
        dml_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.LinearDML",
            control_value=0,
            treatment_value=1,
            target_units=lambda df: df["X0"] > 1,  # condition used for CATE
            method_params={"init_params": {
                'model_y': GradientBoostingRegressor(),
                'model_t': GradientBoostingRegressor(),
                'featurizer': PolynomialFeatures(degree=1, include_bias=True)},
                "fit_params": {}
            }
        )
        # Test ContinuousTreatmentOrthoForest
        orthoforest_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.ortho_forest.ContinuousTreatmentOrthoForest",
            target_units=lambda df: df["X0"] > 2,
            method_params={
                "init_params": {'n_trees': 10},
                "fit_params": {}
            }
        )
        # Test LinearDRLearner
        data_binary = datasets.linear_dataset(
            10, num_common_causes=4, num_samples=10000,
            num_instruments=2, num_effect_modifiers=2,
            treatment_is_binary=True, outcome_is_binary=True)
        model_binary = CausalModel(
            data=data_binary["df"],
            treatment=data_binary["treatment_name"],
            outcome=data_binary["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data_binary["gml_graph"])
        identified_estimand_binary = model_binary.identify_effect(
            proceed_when_unidentifiable=True)
        drlearner_estimate = model_binary.estimate_effect(
            identified_estimand_binary,
            method_name="backdoor.econml.drlearner.LinearDRLearner",
            target_units=lambda df: df["X0"] > 1,
            confidence_intervals=False,
            method_params={
                "init_params": {'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                "fit_params": {}
            })

    def test_iv_estimators(self):
        keras = pytest.importorskip("keras")
        # Setup data
        data = datasets.linear_dataset(
            10, num_common_causes=4, num_samples=10000,
            num_instruments=2, num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=False)
        df = data['df']
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"],
            identify_vars=True
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True)
        # Test DeepIV
        dims_zx = len(model._instruments)+len(model._effect_modifiers)
        dims_tx = len(model._treatment)+len(model._effect_modifiers)
        treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(dims_zx,)),  # sum of dims of Z and X
                                            keras.layers.Dropout(0.17),
                                            keras.layers.Dense(
                                                64, activation='relu'),
                                            keras.layers.Dropout(0.17),
                                            keras.layers.Dense(
                                                32, activation='relu'),
                                            keras.layers.Dropout(0.17)])
        response_model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(
                dims_tx,)),  # sum of dims of T and X
            keras.layers.Dropout(0.17),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.17),
            keras.layers.Dense(1)])
        deepiv_estimate = model.estimate_effect(identified_estimand,
                                                method_name="iv.econml.deepiv.DeepIVEstimator",
                                                target_units=lambda df: df["X0"] > -1,
                                                confidence_intervals=False,
                                                method_params={
                                                    "init_params": {'n_components': 10,  # Number of gaussians in the mixture density networks
                                                                    # Treatment model,
                                                                    'm': lambda z, x: treatment_model(keras.layers.concatenate([z, x])),
                                                                    # Response model
                                                                    "h": lambda t, x: response_model(keras.layers.concatenate([t, x])),
                                                                    'n_samples': 1,  # Number of samples used to estimate the response
                                                                    'first_stage_options': {'epochs': 25},
                                                                    'second_stage_options': {'epochs': 25}
                                                                    },
                                                    "fit_params": {}
                                                }
                                                )
        # Test IntentToTreatDRIV
        data = datasets.linear_dataset(
            10, num_common_causes=4, num_samples=10000,
            num_instruments=1, num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=True,
            num_discrete_instruments=1)
        df = data['df']
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"]
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True)
        driv_estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.econml.ortho_iv.LinearIntentToTreatDRIV",
            target_units=lambda df: df["X0"] > 1,
            confidence_intervals=False,
            method_params={
                "init_params": {'model_t_xwz': GradientBoostingClassifier(),
                                'model_y_xw': GradientBoostingRegressor(),
                                'flexible_model_effect': GradientBoostingRegressor(),
                                'featurizer': PolynomialFeatures(degree=1, include_bias=False)
                                },
                "fit_params": {}})
