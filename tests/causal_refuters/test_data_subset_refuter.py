import numpy as np
from pytest import mark

import dowhy.datasets
from dowhy import CausalModel

from .base import SimpleRefuter


def _build_model_and_estimate(num_samples=2000, seed=0, num_instruments=1):
    """Helper: build a simple linear model and return (model, estimand, estimate)."""
    np.random.seed(seed)
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=1,
        num_instruments=num_instruments,
        num_samples=num_samples,
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
    estimand = model.identify_effect(method_name="exhaustive-search")
    estimand.set_identifier_method("backdoor")
    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name="backdoor.linear_regression",
        test_significance=None,
    )
    return model, estimand, estimate


@mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    @mark.parametrize(["error_tolerance", "estimator_method"], [(0.01, "iv.instrumental_variable")])
    def test_refutation_data_subset_refuter_continuous(self, error_tolerance, estimator_method):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "data_subset_refuter")
        refuter_tester.continuous_treatment_testsuite()  # Run both

    @mark.parametrize(["error_tolerance", "estimator_method"], [(0.01, "backdoor.propensity_score_matching")])
    def test_refutation_data_subset_refuter_binary(self, error_tolerance, estimator_method):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "data_subset_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    def test_random_state_produces_reproducible_results(self):
        """Two runs with the same random_state integer must produce the same refutation."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=500)

        ref1 = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            num_simulations=10,
            random_state=42,
        )
        ref2 = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            num_simulations=10,
            random_state=42,
        )
        assert ref1.new_effect == ref2.new_effect

    def test_random_state_produces_different_simulations(self):
        """With a fixed random_state each simulation must use a DIFFERENT data subset.

        If all simulations used the same subset (the old bug), every call to
        _refute_once would return the same value regardless of the seed passed.
        After the fix, different integer seeds must produce different data samples
        and therefore (virtually always) different estimates.
        """
        from dowhy.causal_refuters.data_subset_refuter import _refute_once

        np.random.seed(0)
        data_dict = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=1000,
            treatment_is_binary=True,
        )
        df = data_dict["df"]
        model = CausalModel(
            data=df,
            treatment=data_dict["treatment_name"],
            outcome=data_dict["outcome_name"],
            graph=data_dict["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        estimand = model.identify_effect(method_name="exhaustive-search")
        estimand.set_identifier_method("backdoor")
        estimate = model.estimate_effect(
            identified_estimand=estimand,
            method_name="backdoor.linear_regression",
            test_significance=None,
        )

        # Call _refute_once with three different integer seeds
        val_a = _refute_once(df, estimand, estimate, 0.8, 1)
        val_b = _refute_once(df, estimand, estimate, 0.8, 2)
        val_c = _refute_once(df, estimand, estimate, 0.8, 3)

        # All three must not be identical (with n=1000 and beta=10, collision is impossible)
        assert not (val_a == val_b == val_c), (
            "All _refute_once calls with different seeds returned the same value — "
            "seeding per-simulation is not working correctly."
        )

    def test_subset_fraction_affects_subset_size(self):
        """Smaller subset_fraction should not degrade the estimate drastically."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=2000)

        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.5,
            num_simulations=5,
            random_state=7,
        )
        # The refuted effect should be in the same ballpark as the original
        assert abs(ref.new_effect - estimate.value) < 5.0

    def test_different_random_states_give_different_results(self):
        """Two different integer random_states should (almost certainly) give different new_effects."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=500)

        ref_a = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            num_simulations=10,
            random_state=1,
        )
        ref_b = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            num_simulations=10,
            random_state=999,
        )
        # It is astronomically unlikely that two independent seeds give the same mean
        assert ref_a.new_effect != ref_b.new_effect
