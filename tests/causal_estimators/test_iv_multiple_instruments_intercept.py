import numpy as np
from pytest import mark

import dowhy.datasets
from dowhy import CausalModel


def _iv_estimate(df, data):
    model = CausalModel(
        data=df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
        proceed_when_unidentifiable=True,
        test_significance=None,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model.estimate_effect(
        estimand,
        method_name="iv.instrumental_variable",
        control_value=0,
        treatment_value=1,
    ).value


@mark.usefixtures("fixed_seed")
def test_multiple_instruments_are_invariant_to_an_outcome_shift():
    """With 2+ instruments the estimator uses 2SLS, which needs an intercept.

    Shifting the outcome by a constant does not change the causal effect of the
    treatment on the outcome. Without an intercept the 2SLS fit is forced through
    the origin, so the shift leaks into the estimate.
    """
    beta = 10
    data = dowhy.datasets.linear_dataset(
        beta=beta,
        num_common_causes=1,
        num_instruments=2,
        num_treatments=1,
        num_samples=20000,
        treatment_is_binary=False,
    )
    df = data["df"]
    outcome = data["outcome_name"][0]

    baseline = _iv_estimate(df, data)
    assert np.isclose(baseline, beta, rtol=0.1)

    df_shifted = df.copy()
    df_shifted[outcome] = df_shifted[outcome] + 100.0
    shifted = _iv_estimate(df_shifted, data)

    assert np.isfinite(shifted)
    assert np.isclose(shifted, baseline, rtol=1e-6)
