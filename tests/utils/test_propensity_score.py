"""Tests for dowhy.utils.propensity_score."""

import numpy as np
import pandas as pd

from dowhy.utils.propensity_score import categorical_treatment_model


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
