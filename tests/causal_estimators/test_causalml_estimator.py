import pytest
import subprocess
import sys

from dowhy import CausalModel
from dowhy.datasets import linear_dataset

# To skip the tests if we cannot install causalml
installed_failed = False
# Hack to install causalml if not already present
try:
    __import__("causalml")
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "causalml"])
        __import__("causalml")
        # We import this later as we obtained this from causalml
        from xgboost import XGBRegressor
    except subprocess.CalledProcessError:
        installed_failed = True

class TestCausalMLEstimator:
    '''
        To test the basic functionality of the CauslML estimators
    '''
    
    @pytest.mark.skipif(installed_failed)
    def test_LRSRegressor(self):
        # Defined a linear dataset with a given set of properties
        data = linear_dataset(
            beta=10,
            num_common_causes=4,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True
        )

        # Create a model that captures the same
        model = CausalModel(
            data=data['df'],
            treatment=data['treatment_name'],
            outcome=data['outcome_name'],
            effect_modifiers=data['effect_modifier_names'],
            graph=data['gml_graph']
        )

        # Identify the effects within the model
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        lr_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.causalml.inference.meta.LRSRegressor",
            method_params={"init_params":{}}
        )

        print("The LR estimate obtained:")
        print(lr_estimate)

    @pytest.mark.skipif(installed_failed)
    def test_XGBTRegressor(self):
        # Defined a linear dataset with a given set of properties
        data = linear_dataset(
            beta=10,
            num_common_causes=4,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True
        )

        # Create a model that captures the same
        model = CausalModel(
            data=data['df'],
            treatment=data['treatment_name'],
            outcome=data['outcome_name'],
            effect_modifiers=data['effect_modifier_names'],
            graph=data['gml_graph']
        )

        # Identify the effects within the model
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        xgbt_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.causalml.inference.meta.XGBTRegressor",
            method_params={"init_params":{}}
        )

        print("The XGBT estimate obtained:")
        print(xgbt_estimate)

    @pytest.mark.skipif(installed_failed)
    def test_MLPTRegressor(self):
        # Defined a linear dataset with a given set of properties
        data = linear_dataset(
            beta=10,
            num_common_causes=4,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True
        )

        # Create a model that captures the same
        model = CausalModel(
            data=data['df'],
            treatment=data['treatment_name'],
            outcome=data['outcome_name'],
            effect_modifiers=data['effect_modifier_names'],
            graph=data['gml_graph']
        )

        # Identify the effects within the model
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        mlpt_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.causalml.inference.meta.MLPTRegressor",
            method_params={"init_params":{
                    'hidden_layer_sizes':(10,10),
                    'learning_rate_init':0.1,
                    'early_stopping':True 
                }
            }
        )

        print("The MLPT estimate obtained:")
        print(mlpt_estimate)

    @pytest.mark.skipif(installed_failed)
    def test_XLearner(self):
        # Defined a linear dataset with a given set of properties
        data = linear_dataset(
            beta=10,
            num_common_causes=4,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True
        )

        # Create a model that captures the same
        model = CausalModel(
            data=data['df'],
            treatment=data['treatment_name'],
            outcome=data['outcome_name'],
            effect_modifiers=data['effect_modifier_names'],
            graph=data['gml_graph']
        )

        # Identify the effects within the model
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        xl_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.causalml.inference.meta.BaseXRegressor",
            method_params={"init_params":{
                    'learner':XGBRegressor()
                }
            }
        )

        print("The X Learner estimate obtained:")
        print(xl_estimate)

    @pytest.mark.skipif(installed_failed)
    def test_RLearner(self):
        # Defined a linear dataset with a given set of properties
        data = linear_dataset(
            beta=10,
            num_common_causes=4,
            num_instruments=2,
            num_effect_modifiers=2,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True
        )

        # Create a model that captures the same
        model = CausalModel(
            data=data['df'],
            treatment=data['treatment_name'],
            outcome=data['outcome_name'],
            effect_modifiers=data['effect_modifier_names'],
            graph=data['gml_graph']
        )

        # Identify the effects within the model
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        rl_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.causalml.inference.meta.BaseRRegressor",
            method_params={"init_params":{
                    'learner':XGBRegressor()
                }
            }
        )

        print("The R Learner estimate obtained:")
        print(rl_estimate)