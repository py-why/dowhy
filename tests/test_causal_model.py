import pytest

import dowhy.datasets
from dowhy import CausalModel


class TestCausalModel(object):
    @pytest.mark.parametrize(["beta", "num_instruments", "num_samples", "num_treatments"],
                             [(10, 1, 100, 1),])
    def test_graph_input(self, beta, num_instruments, num_samples, num_treatments):
        num_common_causes = 5
        data = dowhy.datasets.linear_dataset(beta=beta,
                                             num_common_causes=num_common_causes,
                                             num_instruments=num_instruments,
                                             num_samples=num_samples,
                                             num_treatments = num_treatments,
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
        gml_str = 'graph[directed 1 node[ id "{0}" label "{0}"]node[ id "{1}" label "{1}"]node[ id "Unobserved Confounders" label "Unobserved Confounders"]edge[source "{0}" target "{1}"]edge[source "Unobserved Confounders" target "{0}"]edge[source "Unobserved Confounders" target "{1}"]node[ id "X0" label "X0"] edge[ source "X0" target "{0}"] node[ id "X1" label "X1"] edge[ source "X1" target "{0}"] node[ id "X2" label "X2"] edge[ source "X2" target "{0}"] edge[ source "X0" target "{1}"] edge[ source "X1" target "{1}"] edge[ source "X2" target "{1}"] node[ id "Z0" label "Z0"] edge[ source "Z0" target "{0}"]]'.format(data["treatment_name"][0], data["outcome_name"])
        print(gml_str)
        model = CausalModel(
            data=data['df'],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=gml_str,
            proceed_when_unidentifiable=True,
            test_significance=None,
            missing_nodes_as_confounders=True
        )
        common_causes = model.get_common_causes()
        assert all(node_name in common_causes for node_name in ["X1", "X2"])
