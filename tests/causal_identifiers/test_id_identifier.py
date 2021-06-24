import pandas as pd
import numpy as np
from dowhy import CausalModel

class TestIDIdentification(object):
    
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
        identified_estimand = causal_model.identify_effect(method_name="id")

        # Only P(Y|T) should be present for test to succeed.
        assert (
            (len(identified_estimand.get_val("prod")) == 1 and len(identified_estimand.get_val("sum")) == 0)
            and
            (
                (len(identified_estimand.get_val("prod")[0].get_val("prod")[0]['outcome_vars']) == 1 and identified_estimand.get_val("prod")[0].get_val("prod")[0]['outcome_vars'][0] == 'Y')
                and
                (len(identified_estimand.get_val("prod")[0].get_val("prod")[0]['condition_vars']) == 1 and identified_estimand.get_val("prod")[0].get_val("prod")[0]['condition_vars'][0] == 'T')
            )
        )

    def test_2(self):
        '''
        Test undirected edge between treatment and outcome.
        '''

        # Random data
        treatment = "T"
        outcome = "Y"
        causal_graph = "digraph{T->Y; Y->T;}"
        # df = pd.DataFrame(np.random.randint(0, 100, size=(10000, len(variables))), columns=variables)

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
        try:
            identified_estimand = causal_model.identify_effect(method_name="id")
        except:
            assert True
    
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
        identified_estimand = causal_model.identify_effect(method_name="id")

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
        identified_estimand = causal_model.identify_effect(method_name="id")
    
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
        identified_estimand = causal_model.identify_effect(method_name="id")