import numpy as np
import pandas as pd
import pytest
from numpy.core.fromnumeric import var

from dowhy import CausalModel


class TestIDIdentifier(object):
    def test_1(self):
        treatment = "T"
        outcome = "Y"
        causal_graph = "digraph{T->Y;}"
        columns = list(treatment) + list(outcome)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Only P(Y|T) should be present for test to succeed.
        identified_str = identified_estimand.__str__()
        gt_str = "Predictor: P(Y|T)"
        assert identified_str == gt_str

    def test_2(self):
        """
        Test undirected edge between treatment and outcome.
        """
        treatment = "T"
        outcome = "Y"
        causal_graph = "digraph{T->Y; Y->T;}"
        columns = list(treatment) + list(outcome)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Since undirected graph, identify effect must throw an error.
        with pytest.raises(Exception):
            identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

    def test_3(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T->X1;X1->Y;}"
        columns = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1|T)\n\tPredictor: P(Y|T,X1)"
        assert identified_str == gt_str

    def test_4(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T->Y;T->X1;X1->Y;}"
        columns = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(Y|T,X1)\n\tPredictor: P(X1|T)"
        assert identified_str == gt_str

    def test_5(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2"]
        causal_graph = "digraph{T->Y;X1->T;X1->Y;X2->T;}"
        columns = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        set_a = set(identified_estimand._product[0]._product[0]._product[0]["outcome_vars"]._set)
        set_b = set(identified_estimand._product[0]._product[0]._product[0]["condition_vars"]._set)
        set_c = set(identified_estimand._product[0]._product[1]._product[0]["outcome_vars"]._set)
        set_d = set(identified_estimand._product[0]._product[1]._product[0]["condition_vars"]._set)
        assert identified_estimand._product[0]._sum == ["X1"]
        assert len(set_a.difference({"Y"})) == 0
        assert len(set_b.difference({"X1", "X2", "T"})) == 0
        assert len(set_c.difference({"X1"})) == 0
        assert len(set_d) == 0

    def test_6(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1"]
        causal_graph = "digraph{T;X1->Y;}"
        columns = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=columns)

        # Calculate causal effect twice: once for unit (t=1, c=0), once for specific increase (t=100, c=50)
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)
        identified_estimand = causal_model.identify_effect(method_name="id-algorithm")

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1,Y)"
        assert identified_str == gt_str
