import pytest

from dowhy import identify_effect_id
from dowhy.graph import build_graph_from_str


class TestIDIdentifier(object):
    def test_1(self):
        identified_estimand = identify_effect_id(
            build_graph_from_str("digraph{T->Y;}"), action_nodes=["T"], outcome_nodes=["Y"]
        )

        # Only P(Y|T) should be present for test to succeed.
        identified_str = identified_estimand.__str__()
        gt_str = "Predictor: P(Y|T)"
        assert identified_str == gt_str

    def test_2(self):
        """
        Test undirected edge between treatment and outcome.
        """
        # Since undirected graph, identify effect must throw an error.
        with pytest.raises(Exception):
            identified_estimand = identify_effect_id(
                build_graph_from_str("digraph{T->Y; Y->T;}"), action_nodes=["T"], outcome_nodes=["Y"]
            )

    def test_3(self):
        identified_estimand = identify_effect_id(
            build_graph_from_str("digraph{T->X1;X1->Y;}"), action_nodes=["T"], outcome_nodes=["Y"]
        )

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1|T)\n\tPredictor: P(Y|T,X1)"
        assert identified_str == gt_str

    def test_4(self):
        identified_estimand = identify_effect_id(
            build_graph_from_str("digraph{T->Y;T->X1;X1->Y;}"), action_nodes=["T"], outcome_nodes=["Y"]
        )

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(Y|T,X1)\n\tPredictor: P(X1|T)"
        assert identified_str == gt_str

    def test_5(self):
        identified_estimand = identify_effect_id(
            build_graph_from_str("digraph{T->Y;X1->T;X1->Y;X2->T;}"), action_nodes=["T"], outcome_nodes=["Y"]
        )

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
        identified_estimand = identify_effect_id(
            build_graph_from_str("digraph{T;X1->Y;}"), action_nodes=["T"], outcome_nodes=["Y"]
        )

        # Compare with ground truth
        identified_str = identified_estimand.__str__()
        gt_str = "Sum over {X1}:\n\tPredictor: P(X1,Y)"
        assert identified_str == gt_str
