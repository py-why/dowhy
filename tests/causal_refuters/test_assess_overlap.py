import numpy as np
import pytest

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_refuters.assess_overlap_overrule import OverruleAnalyzer


class TestAssessOverlapRefuter(object):
    @pytest.mark.parametrize(
        "method_name",
        [("assess_overlap")],
    )
    def test_rules(self, method_name):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=10,
            stddev_outcome_noise=5,
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

        refute = model.refute_estimate(identified_estimand, estimate, method_name=method_name)

        # This is a dummy result for now, until we have real output
        assert refute.rules == "Men over 60 years old"
