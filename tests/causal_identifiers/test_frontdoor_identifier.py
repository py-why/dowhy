import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment
from dowhy.causal_identifier.auto_identifier import identify_frontdoor
from dowhy.causal_identifier.identify_effect import EstimandType

from .base import IdentificationTestFrontdoorGraphSolution, example_frontdoor_graph_solution


class TestFrontdoorIdentification(object):
    def test_identify_frontdoor_functional_api(
        self, example_frontdoor_graph_solution: IdentificationTestFrontdoorGraphSolution
    ):
        graph = example_frontdoor_graph_solution.graph
        expected_sets = example_frontdoor_graph_solution.valid_frontdoor_sets
        invalid_sets = example_frontdoor_graph_solution.invalid_frontdoor_sets
        frontdoor_set = identify_frontdoor(
            graph,
            observed_nodes=example_frontdoor_graph_solution.observed_nodes,
            action_nodes=example_frontdoor_graph_solution.action_nodes,
            outcome_nodes=example_frontdoor_graph_solution.outcome_nodes,
        )

        assert (
            (len(frontdoor_set) == 0) and (len(expected_sets) == 0)
        ) or (  # No adjustments exist and that's expected.
            set(frontdoor_set) in expected_sets and set(frontdoor_set) not in invalid_sets
        )

    def test_identify_frontdoor_causal_model(
        self, example_frontdoor_graph_solution: IdentificationTestFrontdoorGraphSolution
    ):
        graph = example_frontdoor_graph_solution.graph
        expected_sets = example_frontdoor_graph_solution.valid_frontdoor_sets
        invalid_sets = example_frontdoor_graph_solution.invalid_frontdoor_sets
        observed_nodes = example_frontdoor_graph_solution.observed_nodes
        # Building the causal model
        num_samples = 10
        df = pd.DataFrame(np.random.random((num_samples, len(observed_nodes))), columns=observed_nodes)
        model = CausalModel(
            data=df,
            treatment=example_frontdoor_graph_solution.action_nodes,
            outcome=example_frontdoor_graph_solution.outcome_nodes,
            graph=graph,
        )
        estimand = model.identify_effect()
        frontdoor_set = estimand.frontdoor_variables
        assert (
            (len(frontdoor_set) == 0) and (len(expected_sets) == 0)
        ) or (  # No adjustments exist and that's expected.
            (set(frontdoor_set) in expected_sets) and (set(frontdoor_set) not in invalid_sets)
        )
