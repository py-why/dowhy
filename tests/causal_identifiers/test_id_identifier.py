import pytest
import pandas as pd
import numpy as np
from dowhy import CausalModel

class TestIDIdentifier(object):
    
    def test_1(self):
        # Random data
        treatment = "T"
        outcome = "Y"
        causal_graph = "digraph{T->Y;}"
        
        # T 
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, 1)), columns=[treatment])

        # Y = 2*T 
        df[outcome] = df[treatment]*2

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Only P(Y|T) should be present for test to succeed.
        identified_str = identified_estimand.__str__()
        gt_str = "Predictor: P(Y|T)"
        assert identified_str == gt_str

    def test_2(self):
        '''
        Test undirected edge between treatment and outcome.
        '''

        # Random data
        treatment = "T"
        outcome = "Y"
        causal_graph = "digraph{T->Y; Y->T;}"
        
        # T 
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, 1)), columns=[treatment])

        # Y = 2*T 
        df[outcome] = df[treatment]*2

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Since undirected graph, identify effect must throw an error.
        with pytest.raises(Exception):
            identified_estimand = causal_model.identify_effect(method_name="id-algorithm")
    
    def test_3(self):
        # Random data
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T->X1;X1->Y;}"

        # T
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, 1)), columns=[treatment])

        # X1 = 0.5*T
        df['X1'] = df[treatment]*0.5

        # Y = 2*T + X1
        df[outcome] = df["X1"]*5

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1|T)\n\tPredictor: P(Y|T,X1)"
        assert identified_str == gt_str

    def test_4(self):
        #Random data
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T->Y;T->X1;X1->Y;}"
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, len(variables))), columns=variables)

        # T
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, 1)), columns=[treatment])

        # X1 = 0.5*T
        df['X1'] = df[treatment]*0.5

        # Y = 2*T + X1
        df[outcome] = df[treatment]*2 + df["X1"]

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(Y|T,X1)\n\tPredictor: P(X1|T)"
        assert identified_str == gt_str

    
    def test_5(self):
        # Random data
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2"]
        causal_graph = "digraph{T->Y;X1->T;X1->Y;X2->T;}"
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, len(variables))), columns=variables)

        # T = X1 * X2
        df[treatment] = df["X1"] * df["X2"]

        # Y = 2*T + X1
        df[outcome] = df[treatment]*2 + df["X1"]*3 + df["X1"]

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(Y|X2,X1,T)\n\tPredictor: P(X1)"
        assert identified_str == gt_str

    def test_6(self):
        # Random data
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T;X1->Y;}"

        # T
        df = pd.DataFrame(np.random.randint(0, 100, size=(10000, 1)), columns=[treatment])

        # X1
        df["X1"] = pd.DataFrame(np.random.randint(0, 100, size=(10000, len(variables))), columns=variables)

        # Y = 2*X1
        df[outcome] = df["X1"]*2

        # Add some noise
        noise = np.random.normal(0, 25, 10000)
        df[outcome] = df[outcome] + noise

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1,Y)"
        assert identified_str == gt_str
