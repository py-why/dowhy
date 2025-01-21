import pandas as pd

from dowhy import CausalModel
from dowhy.causal_identifier.backdoor import Backdoor
from dowhy.utils.api import parse_state


class TestOptimizeBackdoorIdentifier(object):
    def test_1(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2"]
        causal_graph = "digraph{X1->T;X2->T;X1->X2;X2->Y;T->Y}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()
        print(backdoor_sets)
        # Check if backdoor sets are valid i.e. if they block all paths between the treatment and the outcome
        backdoor_paths = causal_model._graph.get_backdoor_paths(treatment_name, outcome_name)
        check_set = set(backdoor_sets[0].get_adjustment_variables())
        check = causal_model._graph.check_valid_backdoor_set(
            treatment_name, outcome_name, check_set, backdoor_paths=backdoor_paths, dseparation_algo="naive"
        )
        print(check)
        assert check["is_dseparated"]

    def test_2(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2"]
        causal_graph = "digraph{T->X1;T->X2;X1->X2;X2->Y;T->Y}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()

        assert len(backdoor_sets) == 0

    def test_3(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2", "X3"]
        causal_graph = "digraph{X1->T;X1->X2;Y->X2;X3->T;X3->Y;T->Y}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()

        # Check if backdoor sets are valid i.e. if they block all paths between the treatment and the outcome
        backdoor_paths = causal_model._graph.get_backdoor_paths(treatment_name, outcome_name)
        check_set = set(backdoor_sets[0].get_adjustment_variables())
        check = causal_model._graph.check_valid_backdoor_set(
            treatment_name, outcome_name, check_set, backdoor_paths=backdoor_paths
        )

        assert check["is_dseparated"]

    def test_4(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2"]
        causal_graph = "digraph{T->Y;X1->T;X1->Y;X2->T;}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()

        # Check if backdoor sets are valid i.e. if they block all paths between the treatment and the outcome
        backdoor_paths = causal_model._graph.get_backdoor_paths(treatment_name, outcome_name)
        check_set = set(backdoor_sets[0].get_adjustment_variables())
        check = causal_model._graph.check_valid_backdoor_set(
            treatment_name, outcome_name, check_set, backdoor_paths=backdoor_paths
        )

        assert check["is_dseparated"]

    def test_5(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2", "X3", "X4"]
        causal_graph = "digraph{X1->T;X1->X2;X2->Y;X3->T;X3->X4;X4->Y;T->Y}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()

        # Check if backdoor sets are valid i.e. if they block all paths between the treatment and the outcome
        backdoor_paths = causal_model._graph.get_backdoor_paths(treatment_name, outcome_name)
        check_set = set(backdoor_sets[0].get_adjustment_variables())
        check = causal_model._graph.check_valid_backdoor_set(
            treatment_name, outcome_name, check_set, backdoor_paths=backdoor_paths
        )

        assert check["is_dseparated"]

    def test_6(self):
        treatment = "T"
        outcome = "Y"
        variables = ["X1", "X2", "X3", "X4"]
        causal_graph = "digraph{X1->T;X1->X2;Y->X2;X3->T;X3->X4;X4->Y;T->Y}"

        vars = list(treatment) + list(outcome) + list(variables)
        df = pd.DataFrame(columns=vars)

        treatment_name = parse_state(treatment)
        outcome_name = parse_state(outcome)

        # Causal model initialization
        causal_model = CausalModel(df, treatment, outcome, graph=causal_graph)

        # Obtain backdoor sets
        path = Backdoor(causal_model._graph._graph, treatment_name, outcome_name)
        backdoor_sets = path.get_backdoor_vars()

        # Check if backdoor sets are valid i.e. if they block all paths between the treatment and the outcome
        backdoor_paths = causal_model._graph.get_backdoor_paths(treatment_name, outcome_name)
        check_set = set(backdoor_sets[0].get_adjustment_variables())
        check = causal_model._graph.check_valid_backdoor_set(
            treatment_name, outcome_name, check_set, backdoor_paths=backdoor_paths
        )

        assert check["is_dseparated"]
