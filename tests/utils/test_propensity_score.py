"""Tests for dowhy.utils.propensity_score."""

import numpy as np
import pandas as pd

from dowhy.utils.propensity_score import binary_treatment_model, categorical_treatment_model, continuous_treatment_model


def test_categorical_treatment_model_runs_with_multivalued_treatment():
    """Regression test: the multi-valued treatment propensity model must run on
    modern scikit-learn (the removed ``LogisticRegression(multi_class=...)``
    argument used to raise ``TypeError`` on scikit-learn >= 1.7)."""
    rng = np.random.RandomState(0)
    n = 300
    data = pd.DataFrame({"W": rng.normal(size=n), "T": rng.randint(0, 3, size=n)})
    scores = categorical_treatment_model(data.copy(), ["W"], "T", {"W": "c", "T": "d"})
    assert len(scores) == n
    assert np.all((scores >= 0) & (scores <= 1))


# --- Regression tests for issue #225: empty-covariate path ------------------


def test_binary_treatment_model_no_covariates():
    """binary_treatment_model must not crash with an empty covariate list.

    Previously raised ``ValueError: Found array with 0 feature(s)`` from sklearn.
    """
    rng = np.random.RandomState(0)
    n = 200
    p_treat = 0.6
    data = pd.DataFrame({"T": rng.binomial(1, p_treat, size=n)})
    scores = binary_treatment_model(data.copy(), [], "T", {"T": "b"})
    assert len(scores) == n
    assert np.all((scores >= 0) & (scores <= 1))
    # Treated rows should get p_treat, control rows should get 1-p_treat
    treated_mask = data["T"].values.astype(bool)
    np.testing.assert_allclose(scores[treated_mask], p_treat, atol=0.05)
    np.testing.assert_allclose(scores[~treated_mask], 1.0 - p_treat, atol=0.05)


def test_categorical_treatment_model_no_covariates():
    """categorical_treatment_model must not crash with an empty covariate list."""
    rng = np.random.RandomState(0)
    n = 300
    data = pd.DataFrame({"T": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])})
    scores = categorical_treatment_model(data.copy(), [], "T", {"T": "d"})
    assert len(scores) == n
    assert np.all((scores >= 0) & (scores <= 1))


def test_continuous_treatment_model_no_covariates():
    """continuous_treatment_model must not crash with an empty covariate list."""
    rng = np.random.RandomState(0)
    n = 100
    data = pd.DataFrame({"T": rng.normal(size=n)})
    scores = continuous_treatment_model(data.copy(), [], "T", {"T": "c"})
    assert len(scores) == n
    assert np.all(scores > 0), "Density estimates must be positive"
