import pytest

import dowhy.datasets
from dowhy import CausalModel


class TestCausalModel(object):
    @pytest.mark.parametrize(["beta", "num_instruments", "num_samples"],
                             [(10, 1, 100),])
    def test_graph_input(self, beta, num_instruments, num_samples):
        num_common_causes = 5
        data = dowhy.datasets.linear_dataset(beta=beta,
                                             num_common_causes=num_common_causes,
                                             num_instruments=num_instruments,
                                             num_samples=num_samples,
                                             treatment_is_binary=True)

        model = CausalModel(
            data=data['df'],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None
        )
        # removing two common causes
        gml_str = 'graph[directed 1node[ id "v" label "v"]node[ id "y" label "y"]node[ id "Unobserved Confounders" label "Unobserved Confounders"]edge[source "v" target "y"]edge[source "Unobserved Confounders" target "v"]edge[source "Unobserved Confounders" target "y"]node[ id "X0" label "X0"] edge[ source "X0" target "v"] node[ id "X1" label "X1"] edge[ source "X1" target "v"] node[ id "X2" label "X2"] edge[ source "X2" target "v"] edge[ source "X0" target "y"] edge[ source "X1" target "y"] edge[ source "X2" target "y"] node[ id "Z0" label "Z0"] edge[ source "Z0" target "v"]]'
        model = CausalModel(
            data=data['df'],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=gml_str,
            proceed_when_unidentifiable=True,
            test_significance=None,
            missing_nodes_as_confounders=True
        )
        assert all(node_name in model._common_causes for node_name in ["X1", "X2"])

