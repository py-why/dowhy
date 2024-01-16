import re

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import dowhy
from dowhy import CausalModel, datasets

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
            10,
            num_common_causes=4,
            num_samples=1000,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=False,
        )
        df = data["df"]
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"],
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Test LinearDML
        dml_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.dml.LinearDML",
            control_value=0,
            treatment_value=1,
            target_units=lambda df: df["X0"] > 1,  # condition used for CATE
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": GradientBoostingRegressor(),
                    "featurizer": PolynomialFeatures(degree=1, include_bias=True),
                },
                "fit_params": {},
            },
        )
        # Checking that the CATE estimates are not identical
        dml_cate_estimates_f = dml_estimate.cate_estimates.flatten()
        assert pytest.approx(dml_cate_estimates_f[0], 0.001) != dml_cate_estimates_f[1]
        # Test ContinuousTreatmentOrthoForest
        orthoforest_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.orf.DMLOrthoForest",
            target_units=lambda df: df["X0"] > 2,
            method_params={"init_params": {"n_trees": 10}, "fit_params": {}},
        )
        # Checking that the CATE estimates are not identical
        orthoforest_cate_estimates_f = orthoforest_estimate.cate_estimates.flatten()
        assert pytest.approx(orthoforest_cate_estimates_f[0], 0.001) != orthoforest_cate_estimates_f[1]

        # Test LinearDRLearner
        data_binary = datasets.linear_dataset(
            10,
            num_common_causes=4,
            num_samples=1000,
            num_instruments=2,
            num_effect_modifiers=2,
            treatment_is_binary=True,
            outcome_is_binary=True,
        )
        model_binary = CausalModel(
            data=data_binary["df"],
            treatment=data_binary["treatment_name"],
            outcome=data_binary["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data_binary["gml_graph"],
        )
        identified_estimand_binary = model_binary.identify_effect(proceed_when_unidentifiable=True)
        drlearner_estimate = model_binary.estimate_effect(
            identified_estimand_binary,
            method_name="backdoor.econml.dr.LinearDRLearner",
            target_units=lambda df: df["X0"] > 1,
            confidence_intervals=False,
            method_params={
                "init_params": {"model_propensity": LogisticRegressionCV(cv=3, solver="lbfgs", multi_class="auto")},
                "fit_params": {},
            },
        )
        drlearner_cate_estimates_f = drlearner_estimate.cate_estimates.flatten()
        assert pytest.approx(drlearner_cate_estimates_f[0], 0.001) != drlearner_cate_estimates_f[1]

    def test_metalearners(self):
        data = datasets.linear_dataset(
            10,
            num_common_causes=4,
            num_samples=1000,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=True,
        )
        df = data["df"]
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"],
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Test LinearDML
        sl_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.metalearners.SLearner",
            target_units="ate",
            method_params={"init_params": {"overall_model": GradientBoostingRegressor()}, "fit_params": {}},
        )
        # checking that CATE estimates are not identical
        sl_cate_estimates_f = sl_estimate.cate_estimates.flatten()
        assert pytest.approx(sl_cate_estimates_f[0], 0.01) != sl_cate_estimates_f[1]

        # predict on new data
        sl_estimate_test = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.metalearners.SLearner",
            fit_estimator=False,
            target_units=data["df"].sample(frac=0.1),
        )
        sl_cate_estimates_test_f = sl_estimate_test.cate_estimates.flatten()
        assert pytest.approx(sl_cate_estimates_test_f[0], 0.01) != sl_cate_estimates_test_f[1]

    def test_iv_estimators(self):
        keras = pytest.importorskip("keras")
        # Setup data
        data = datasets.linear_dataset(
            10,
            num_common_causes=4,
            num_samples=1000,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=False,
        )
        df = data["df"]
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"],
            identify_vars=True,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Test DeepIV
        dims_zx = len(model._instruments) + len(model._effect_modifiers)
        dims_tx = len(model._treatment) + len(model._effect_modifiers)
        treatment_model = keras.Sequential(
            [
                keras.layers.Dense(128, activation="relu", input_shape=(dims_zx,)),  # sum of dims of Z and X
                keras.layers.Dropout(0.17),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.17),
            ]
        )
        response_model = keras.Sequential(
            [
                keras.layers.Dense(128, activation="relu", input_shape=(dims_tx,)),  # sum of dims of T and X
                keras.layers.Dropout(0.17),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.17),
                keras.layers.Dense(1),
            ]
        )
        dmliv_estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.econml.iv.dml.DMLIV",
            target_units=lambda df: df["X0"] > -1,
            confidence_intervals=False,
            method_params={
                "init_params": {"discrete_treatment": False, "discrete_instrument": False},
                "fit_params": {},
            },
        )
        # Test IntentToTreatDRIV
        data = datasets.linear_dataset(
            10,
            num_common_causes=4,
            num_samples=1000,
            num_instruments=1,
            num_effect_modifiers=2,
            num_treatments=1,
            treatment_is_binary=True,
            num_discrete_instruments=1,
        )
        df = data["df"]
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            effect_modifiers=data["effect_modifier_names"],
            graph=data["gml_graph"],
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        driv_estimate = model.estimate_effect(
            identified_estimand,
            method_name="iv.econml.iv.dr.LinearIntentToTreatDRIV",
            target_units=lambda df: df["X0"] > 1,
            confidence_intervals=False,
            method_params={
                "init_params": {
                    "model_t_xwz": GradientBoostingClassifier(),
                    "model_y_xw": GradientBoostingRegressor(),
                    "flexible_model_effect": GradientBoostingRegressor(),
                    "featurizer": PolynomialFeatures(degree=1, include_bias=False),
                },
                "fit_params": {},
            },
        )

    def test_multivalue_treatment(self):
        n_points = 100000
        impact = {0: 0.0, 1: 2.0, 2: 1.0}
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=n_points),
                "W": np.random.normal(size=n_points),
                "T": np.random.choice(np.array(list(impact.keys())), size=n_points),
            }
        )
        df["Y"] = df["W"] + df["T"].apply(lambda x: impact[x])

        train_data, test_data = train_test_split(df, train_size=0.9)

        causal_model = CausalModel(
            data=train_data,
            treatment="T",
            outcome="Y",
            common_causes="W",
            effect_modifiers="X",
        )
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)

        est_2 = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.dml.LinearDML",
            control_value=0,
            treatment_value=[1, 2],
            target_units="ate",  # condition used for CATE
            confidence_intervals=False,
            method_params={
                "init_params": {"discrete_treatment": True},
                "fit_params": {},
            },
        )

        est_test = est_2.estimator.effect_tt(test_data, [1, 2])

        est_error = (est_test - test_data["T"].apply(lambda x: impact[x]).values).abs().max()
        assert est_error < 0.03

    def test_empty_effect_modifiers(self):
        np.random.seed(101)
        data = dowhy.datasets.partially_linear_dataset(
            beta=10,
            num_common_causes=7,
            num_unobserved_common_causes=1,
            strength_unobserved_confounding=10,
            num_samples=1000,
            num_treatments=1,
            stddev_treatment_noise=10,
            stddev_outcome_noise=5,
        )

        # Observed data
        dropped_cols = ["W0"]
        user_data = data["df"].drop(dropped_cols, axis=1)
        # assumed graph
        user_graph = data["gml_graph"]
        for col in dropped_cols:
            user_graph = user_graph.replace('node[ id "{0}" label "{0}"]'.format(col), "")
            user_graph = re.sub('edge\[ source "{}" target "[vy][0]*"\]'.format(col), "", user_graph)

        model = CausalModel(
            data=user_data,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=user_graph,
            test_significance=None,
        )

        model._graph.get_effect_modifiers(model._treatment, model._outcome)

        # Identify effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate effect
        model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.dml.LinearDML",
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": GradientBoostingRegressor(),
                    "linear_first_stages": False,
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )

    def test_empty_effect_modifier_nograph(self):
        arr = np.random.normal(size=(500, 5))
        df = pd.DataFrame(arr, columns=["Y", "T", "X", "W0", "W1"])
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est1 = model.estimate_effect(
            identified_estimand=estimand,
            effect_modifiers=[],
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        assert est1.cate_estimates.shape == (1, 1)

        # A model where X is also a common cause
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["X", "W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est2 = model.estimate_effect(
            identified_estimand=estimand,
            effect_modifiers=[],
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        assert est2.cate_estimates.shape == (1, 1)

    def test_effect_modifier_input(self):
        arr = np.random.normal(size=(500, 5))
        df = pd.DataFrame(arr, columns=["Y", "T", "X", "W0", "W1"])
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["X", "W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est1 = model.estimate_effect(
            identified_estimand=estimand,
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        # A model where X is also a common cause
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["X", "W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est2 = model.estimate_effect(
            identified_estimand=estimand,
            effect_modifiers=["X"],
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        assert np.array_equal(est1.cate_estimates, est2.cate_estimates)

    def test_effect_modifier_input2(self):
        # The case where effect modifier is not a common cause
        arr = np.random.normal(size=(500, 5))
        df = pd.DataFrame(arr, columns=["Y", "T", "X", "W0", "W1"])
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est1 = model.estimate_effect(
            identified_estimand=estimand,
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        # A model where X is also a common cause
        model = CausalModel(
            data=df,
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"],
            common_causes=["W0", "W1"],
            estimand_type="nonparametric-ate",
        )

        estimand = model.identify_effect()
        est2 = model.estimate_effect(
            identified_estimand=estimand,
            effect_modifiers=["X"],
            method_name="backdoor.econml.dml.LinearDML",
            method_params={"init_params": {"random_state": 123}, "fit_params": {}},
        )

        assert np.array_equal(est1.cate_estimates, est2.cate_estimates)
