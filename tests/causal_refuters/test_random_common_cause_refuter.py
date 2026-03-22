import numpy as np
import pytest

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_refuters.random_common_cause import refute_random_common_cause

from .base import SimpleRefuter


@pytest.mark.usefixtures("fixed_seed")
class TestRandomCommonCauseRefuter:
    @pytest.mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"],
        [(0.1, "backdoor.linear_regression", 1000)],
    )
    def test_refutation_random_common_cause_continuous(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "random_common_cause")
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples)

    @pytest.mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"],
        [(0.1, "backdoor.linear_regression", 5000)],
    )
    def test_refutation_random_common_cause_binary(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "random_common_cause")
        refuter_tester.binary_treatment_testsuite(
            tests_to_run="atleast-one-common-cause", num_samples=num_samples
        )

    @pytest.mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"],
        [(0.1, "backdoor.linear_regression", 5000)],
    )
    def test_refutation_random_common_cause_category(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "random_common_cause")
        refuter_tester.categorical_treatment_testsuite(
            tests_to_run="atleast-one-common-cause", num_samples=num_samples
        )

    def test_refutation_random_common_cause_refutation_type(self):
        """Test that the refutation type string is set correctly."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        target_estimand = model.identify_effect(method_name="exhaustive-search")
        target_estimand.set_identifier_method("backdoor")
        ate_estimate = model.estimate_effect(
            identified_estimand=target_estimand,
            method_name="backdoor.linear_regression",
            test_significance=None,
        )
        ref = model.refute_estimate(
            target_estimand, ate_estimate, method_name="random_common_cause", num_simulations=10
        )
        assert "random common cause" in ref.refutation_type.lower()

    def test_refute_random_common_cause_functional_api(self):
        """Test the functional API refute_random_common_cause directly."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        target_estimand = model.identify_effect(method_name="exhaustive-search")
        target_estimand.set_identifier_method("backdoor")
        ate_estimate = model.estimate_effect(
            identified_estimand=target_estimand,
            method_name="backdoor.linear_regression",
            test_significance=None,
        )
        ref = refute_random_common_cause(
            data=data["df"],
            target_estimand=target_estimand,
            estimate=ate_estimate,
            num_simulations=10,
            random_state=42,
        )
        assert ref is not None
        assert ref.new_effect is not None
        assert np.isfinite(ref.new_effect)

    def test_refute_random_common_cause_reproducible_with_random_state(self):
        """Test that using an integer random_state gives reproducible results."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        target_estimand = model.identify_effect(method_name="exhaustive-search")
        target_estimand.set_identifier_method("backdoor")
        ate_estimate = model.estimate_effect(
            identified_estimand=target_estimand,
            method_name="backdoor.linear_regression",
            test_significance=None,
        )
        ref1 = refute_random_common_cause(
            data=data["df"],
            target_estimand=target_estimand,
            estimate=ate_estimate,
            num_simulations=5,
            random_state=0,
        )
        ref2 = refute_random_common_cause(
            data=data["df"],
            target_estimand=target_estimand,
            estimate=ate_estimate,
            num_simulations=5,
            random_state=0,
        )
        assert ref1.new_effect == pytest.approx(ref2.new_effect)
