import networkx as nx
import pandas as pd
import pytest
from flaky import flaky
from pytest import mark

import dowhy
import dowhy.datasets
from dowhy import CausalModel
from dowhy.graph import *
from dowhy.utils.graph_operations import daggity_to_dot


class TestCausalGraph(object):
    @pytest.fixture(autouse=True)
    def _init_graph(self):
        self.daggity_file = "tests/sample_dag.txt"
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            # num_frontdoor_variables=1,
            num_effect_modifiers=3,
            num_samples=100,
            num_treatments=1,
            treatment_is_binary=True,
        )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=self.daggity_file,
            proceed_when_unidentifiable=True,
            test_significance=None,
            missing_nodes_as_confounders=False,
        )
        self.graph_obj = model._graph

        # creating nx graph instance
        with open(self.daggity_file, "r") as text_file:
            graph_str = text_file.read()
        graph_str = daggity_to_dot(graph_str)
        # to be used later for a test. Does not include the replace operation
        self.graph_str = graph_str
        graph_str = graph_str.replace("\n", " ")

        import pygraphviz as pgv

        nx_graph = pgv.AGraph(graph_str, strict=True, directed=True)
        nx_graph = nx.drawing.nx_agraph.from_agraph(nx_graph)
        self.nx_graph = nx_graph
        self.action_node = data["treatment_name"]
        self.outcome_node = data["outcome_name"]
        self.observed_nodes = list(nx_graph.nodes)
        self.observed_nodes.remove("Unobserved Confounders")

    def test_check_valid_backdoor_set(self):
        res1 = self.graph_obj.check_valid_backdoor_set(self.action_node, self.outcome_node, ["X1", "X2"])
        res2 = check_valid_backdoor_set(self.nx_graph, self.action_node, self.outcome_node, ["X1", "X2"])
        assert res1 == res2

    def test_do_surgery(self):
        res1 = self.graph_obj.do_surgery(self.action_node)
        res2 = do_surgery(self.nx_graph, self.action_node)
        assert list(res1.nodes) == list(res2.nodes)
        assert res1.edges == res2.edges

    def test_get_backdoor_paths(self):
        res1 = self.graph_obj.get_backdoor_paths(self.action_node, self.outcome_node)
        res2 = get_backdoor_paths(self.nx_graph, self.action_node, self.outcome_node)
        assert res1 == res2

    def test_check_dseparation(self):
        res1 = self.graph_obj.check_dseparation(self.action_node, self.outcome_node, ["X1", "X2"])
        res2 = check_dseparation(self.nx_graph, self.action_node, self.outcome_node, ["X1", "X2"])
        assert res1 == res2

    def test_get_instruments(self):
        res1 = self.graph_obj.get_instruments(self.action_node, self.outcome_node)
        res2 = get_instruments(self.nx_graph, self.action_node, self.outcome_node)
        assert res1 == res2

    def test_get_all_nodes(self):
        for flag in [True, False]:
            print(list(self.graph_obj._graph.nodes))
            print(list(self.nx_graph.nodes))
            res1 = self.graph_obj.get_all_nodes(include_unobserved=flag)
            res2 = get_all_nodes(self.nx_graph, self.observed_nodes, include_unobserved_nodes=flag)
            assert set(res1) == set(res2)

    def test_valid_frontdoor_set(self):
        res1 = self.graph_obj.check_valid_frontdoor_set(self.action_node, self.outcome_node, ["X0"])
        res2 = check_valid_frontdoor_set(self.nx_graph, self.action_node, self.outcome_node, ["X0"])
        assert res1 == res2

    def test_valid_mediation_set(self):
        res1 = self.graph_obj.check_valid_mediation_set(self.action_node, self.outcome_node, ["X0"])
        res2 = check_valid_mediation_set(self.nx_graph, self.action_node, self.outcome_node, ["X0"])
        assert res1 == res2

    def test_build_graph(self):
        data = dowhy.datasets.linear_dataset(beta=10, num_common_causes=1, num_instruments=1, num_samples=100)
        res1 = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            common_causes=["W0"],
            instruments=["Z0"],
            missing_nodes_as_confounders=False,
        )._graph._graph
        res2 = build_graph(
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            common_cause_nodes=["W0"],
            instrument_nodes=["Z0"],
        )
        assert res1.edges == res2.edges

    def test_build_graph_from_str(self):
        build_graph_from_str(self.daggity_file)
        build_graph_from_str(self.graph_str)

    def test_has_path(self):
        assert has_directed_path(self.nx_graph, ["X0"], ["y"])
        assert has_directed_path(self.nx_graph, ["X0", "X1", "X2"], ["y", "v0"])
        assert not has_directed_path(self.nx_graph, [], ["y"])
        assert not has_directed_path(self.nx_graph, ["X0", "X1", "X2"], ["y", "v0", "Z0"])
