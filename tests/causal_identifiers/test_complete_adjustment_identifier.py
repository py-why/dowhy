import sys

import pytest

from dowhy.causal_identifier import GeneralizedAdjustment
from dowhy.causal_identifier.auto_identifier import identify_generalized_adjustment_set

from .base import IdentificationTestGeneralCovariateAdjustmentGraphSolution, example_complete_adjustment_graph_solution


class TestGeneralAdjustmentIdentification(object):
    def test_identify_minimal_adjustment(
        self, example_complete_adjustment_graph_solution: IdentificationTestGeneralCovariateAdjustmentGraphSolution
    ):
        if sys.version_info < (3, 10):  # generalized adjustment identification requires python >=3.10
            return

        graph = example_complete_adjustment_graph_solution.graph
        expected_sets = example_complete_adjustment_graph_solution.minimal_adjustment_sets
        adjustment_set_results = identify_generalized_adjustment_set(
            graph,
            action_nodes=example_complete_adjustment_graph_solution.action_nodes,
            outcome_nodes=example_complete_adjustment_graph_solution.outcome_nodes,
            observed_nodes=example_complete_adjustment_graph_solution.observed_nodes,
            generalized_adjustment=GeneralizedAdjustment.GENERALIZED_ADJUSTMENT_DEFAULT,
        )
        adjustment_sets = [set(adjustment_set.get_adjustment_variables()) for adjustment_set in adjustment_set_results]

        assert all(
            (len(expected_sets) == 0 and len(adjustment_sets) == 0) or set(s) in expected_sets for s in adjustment_sets
        )
