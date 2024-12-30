import pytest

from dowhy.causal_identifier import AutoIdentifier, CovariateAdjustment
from dowhy.causal_identifier.identify_effect import EstimandType

from dowhy.causal_identifier.auto_identifier import identify_complete_adjustment_set

from .base import IdentificationTestGeneralCovariateAdjustmentGraphSolution, example_complete_adjustment_graph_solution


class TestGeneralAdjustmentIdentification(object):
    def test_identify_general_adjustment(self, example_complete_adjustment_graph_solution: IdentificationTestGeneralCovariateAdjustmentGraphSolution):
        graph = example_complete_adjustment_graph_solution.graph
        expected_sets = example_complete_adjustment_graph_solution.exhaustive_adjustment_sets
        adjustment_set_results = identify_complete_adjustment_set(
            graph,
            action_nodes=["X"],
            outcome_nodes=["Y"],
            observed_nodes=example_complete_adjustment_graph_solution.observed_nodes,
            covariate_adjustment=CovariateAdjustment.COVARIATE_ADJUSTMENT_EXHAUSTIVE,
        )
        adjustment_sets = [
            set(adjustment_set.get_variables())
            for adjustment_set in adjustment_set_results
            if len(adjustment_set.get_variables()) > 0
        ]

        assert all(
            (len(s) == 0 and len(adjustment_sets) == 0) or set(s) in adjustment_sets
            for s in expected_sets
        )


