import networkx as nx
import pandas as pd
import pytest
from flaky import flaky
from pytest import mark

import dowhy
import dowhy.datasets
from dowhy import CausalModel
from dowhy.graph import check_valid_backdoor_set, do_surgery, get_backdoor_paths, check_dseparation, get_instruments
from dowhy.utils.graph_operations import daggity_to_dot


class TestCausalGraph(object):

    @pytest.fixture(autouse=True)
    def _init_graph(self):
        daggity_file = "tests/sample_dag.txt"
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_instruments=1,
            num_samples=100,
            num_treatments=1,
            treatment_is_binary=True,
            )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=daggity_file,
            proceed_when_unidentifiable=True,
            test_significance=None,
            missing_nodes_as_confounders=False,
        )
        self.graph_obj = model._graph

        # creating nx graph instance 
        with open(daggity_file, "r") as text_file:
            graph_str = text_file.read()
        graph_str = daggity_to_dot(graph_str)
        graph_str = graph_str.replace("\n", " ")
        
        import pygraphviz as pgv
        nx_graph = pgv.AGraph(graph_str, strict=True, directed=True)
        nx_graph = nx.drawing.nx_agraph.from_agraph(nx_graph)
        self.nx_graph = nx_graph
        self.action_node = data["treatment_name"]
        self.outcome_node = data["outcome_name"]

    def test_check_valid_backdoor_set(self):
        res1 = self.graph_obj.check_valid_backdoor_set(
                self.action_node,
                self.outcome_node,
                ["X1", "X2"])
        res2 = check_valid_backdoor_set(self.nx_graph, 
                self.action_node,
                self.outcome_node, 
                ["X1", "X2"])
        assert res1 == res2

    def test_do_surgery(self):
        res1 = self.graph_obj.do_surgery(self.action_node)
        res2 = do_surgery(self.nx_graph, self.action_node)
        assert list(res1.nodes) == list(res2.nodes) 
        assert res1.edges == res2.edges

    def test_get_backdoor_paths(self):
        res1 = self.graph_obj.get_backdoor_paths(
                self.action_node, self.outcome_node)
        res2 = get_backdoor_paths(self.nx_graph,
                self.action_node, self.outcome_node)
        assert res1 == res2

    def test_check_dseparation(self):
        res1 = self.graph_obj.check_dseparation(
                self.action_node, self.outcome_node, 
                ["X1", "X2"])
        res2 = check_dseparation(self.nx_graph,
                self.action_node, self.outcome_node, 
                ["X1", "X2"])
        assert res1 == res2

    def test_get_instruments(self):
        res1 = self.graph_obj.get_instruments(
                self.action_node, self.outcome_node)
        res2 = get_instruments(self.nx_graph,
                self.action_node, self.outcome_node) 
        assert res1 == res2
