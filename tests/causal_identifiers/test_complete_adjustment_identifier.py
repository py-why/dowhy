import pytest

from dowhy.causal_identifier import AutoIdentifier, CovariateAdjustment
from dowhy.causal_identifier.auto_identifier import identify_complete_adjustment_set
from dowhy.causal_identifier.identify_effect import EstimandType

from .base import IdentificationTestGeneralCovariateAdjustmentGraphSolution, example_complete_adjustment_graph_solution


class TestGeneralAdjustmentIdentification(object):
    def test_identify_minimal_adjustment(
        self, example_complete_adjustment_graph_solution: IdentificationTestGeneralCovariateAdjustmentGraphSolution
    ):
        graph = example_complete_adjustment_graph_solution.graph
        expected_set = example_complete_adjustment_graph_solution.minimal_adjustment_sets[0]
        adjustment_set_results = identify_complete_adjustment_set(
            graph,
            action_nodes=example_complete_adjustment_graph_solution.action_nodes,
            outcome_nodes=example_complete_adjustment_graph_solution.outcome_nodes,
            observed_nodes=example_complete_adjustment_graph_solution.observed_nodes,
            covariate_adjustment=CovariateAdjustment.COVARIATE_ADJUSTMENT_DEFAULT,
        )
        adjustment_sets = [
            set(adjustment_set.get_variables())
            for adjustment_set in adjustment_set_results
            if len(adjustment_set.get_variables()) > 0
        ]

        assert (len(expected_set) == 0 and len(adjustment_sets) == 0) or set(expected_set) in adjustment_sets
