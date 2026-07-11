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
def test_binary_instrument_coding_invariance():
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=1,
        num_instruments=1,
        num_treatments=1,
        num_samples=10000,
        treatment_is_binary=True,
    )
    inst = data["instrument_names"][0]
    df = data["df"]
    assert set(np.unique(df[inst])) == {0, 1}

    baseline = _iv_estimate(df, data)
    assert np.isfinite(baseline)

    df_recoded = df.copy()
    df_recoded[inst] = df_recoded[inst] + 1  # {0,1} -> {1,2}
    recoded = _iv_estimate(df_recoded, data)

    assert np.isfinite(recoded)
    assert np.isclose(recoded, baseline, rtol=1e-6)
