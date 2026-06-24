import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_refuters.iv_exclusion_sensitivity_refuter import IvExclusionSensitivityRefuter


def generate_iv_data(n=1000, num_instruments=1, collinear=False):
    np.random.seed(42)
    U = np.random.normal(0, 1, n)

    if num_instruments == 1:
        Z = np.random.normal(0, 1, n)
        D = 0.8 * Z + 1.5 * U + np.random.normal(0, 0.1, n)
        Y = 2.0 * D + 0.5 * Z + 1.5 * U + np.random.normal(0, 0.1, n)
        return pd.DataFrame({"Z": Z, "D": D, "Y": Y})
    else:
        Z1 = np.random.normal(0, 1, n)
        Z2 = Z1 + np.random.normal(0, 0.0001, n) if collinear else np.random.normal(0, 1, n)
        D = 0.8 * Z1 + 0.4 * Z2 + 1.5 * U + np.random.normal(0, 0.1, n)
        Y = 2.0 * D + 0.5 * Z1 + 0.0 * Z2 + 1.5 * U + np.random.normal(0, 0.1, n)
        return pd.DataFrame({"Z1": Z1, "Z2": Z2, "D": D, "Y": Y})


class TestIvExclusionSensitivityRefuter:

    def get_fitted_model(self, df, instruments):
        model = CausalModel(data=df, treatment="D", outcome="Y", instruments=instruments, common_causes=[])
        ident = model.identify_effect(proceed_when_unidentifiable=True)
        est = model.estimate_effect(ident, method_name="iv.instrumental_variable")
        return model, ident, est

    def test_single_instrument(self):
        df = generate_iv_data(n=2000, num_instruments=1)
        model, ident, est = self.get_fitted_model(df, ["Z"])

        refuter = IvExclusionSensitivityRefuter(
            data=df, identified_estimand=ident, estimate=est, gamma_prior_mean=0.5, gamma_prior_var=0.01
        )
        res = refuter.refute_estimate()
        assert np.isclose(res.new_effect, 2.0, atol=0.1)

    def test_multiple_instruments_dict_mapping(self):
        df = generate_iv_data(n=2000, num_instruments=2)
        model, ident, est = self.get_fitted_model(df, ["Z1", "Z2"])

        refuter = IvExclusionSensitivityRefuter(
            data=df,
            identified_estimand=ident,
            estimate=est,
            gamma_prior_mean={"Z1": 0.5, "Z2": 0.0},
            gamma_prior_var={"Z1": 0.01, "Z2": 0.01},
        )
        res = refuter.refute_estimate()
        assert np.isclose(res.new_effect, 2.0, atol=0.1)

    def test_negative_variance_raises_error(self):
        df = generate_iv_data(n=100, num_instruments=1)
        model, ident, est = self.get_fitted_model(df, ["Z"])

        refuter = IvExclusionSensitivityRefuter(
            data=df, identified_estimand=ident, estimate=est, gamma_prior_mean=0.0, gamma_prior_var=-0.5
        )
        with pytest.raises(ValueError, match="non-negative"):
            refuter.refute_estimate()

    def test_n_less_than_k_raises_error(self):
        # N=2, K=2 (Treatment + Instrument)
        df = generate_iv_data(n=2, num_instruments=1)
        model, ident, est = self.get_fitted_model(df, ["Z"])

        refuter = IvExclusionSensitivityRefuter(
            data=df, identified_estimand=ident, estimate=est, gamma_prior_mean=0.0, gamma_prior_var=0.0
        )
        with pytest.raises(ValueError, match="Degrees of freedom exhausted"):
            refuter.refute_estimate()

    def test_collinear_instruments_warning(self):
        df = generate_iv_data(n=500, num_instruments=2, collinear=True)
        model, ident, est = self.get_fitted_model(df, ["Z1", "Z2"])

        refuter = IvExclusionSensitivityRefuter(
            data=df,
            identified_estimand=ident,
            estimate=est,
            gamma_prior_mean={"Z1": 0.0, "Z2": 0.0},
            gamma_prior_var={"Z1": 0.0, "Z2": 0.0},
        )
        with pytest.warns(UserWarning, match="numerical instability"):
            refuter.refute_estimate()

    def test_zero_variance_collapses_to_standard_error(self):
        df = generate_iv_data(n=1000, num_instruments=1)
        model, ident, est = self.get_fitted_model(df, ["Z"])

        # When gamma_prior_mean=0 and gamma_prior_var=0, the LTZ adjusted
        # standard error should theoretically match the manual 2SLS standard error
        refuter = IvExclusionSensitivityRefuter(
            data=df, identified_estimand=ident, estimate=est, gamma_prior_mean=0.0, gamma_prior_var=0.0
        )
        res = refuter.refute_estimate()
        assert res.new_effect_standard_error > 0