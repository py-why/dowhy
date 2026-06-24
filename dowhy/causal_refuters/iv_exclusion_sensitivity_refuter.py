import warnings

import numpy as np
import scipy.stats as st

from dowhy.causal_refuter import CausalRefutation, CausalRefuter


class IvExclusionSensitivityRefuter(CausalRefuter):
    """
    Refuter for Instrumental Variable estimations that performs an approximate
    Conley-style sensitivity analysis under normally distributed exclusion violations.

    This uses a delta-method variance approximation. For multiple instruments,
    this implementation assumes independent priors (Covariance = 0 between the direct
    effects of different instruments).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma_prior_mean = kwargs.get("gamma_prior_mean", 0.0)
        self.gamma_prior_var = kwargs.get("gamma_prior_var", 0.01)

    def refute_estimate(self, show_progress_bar=False):
        if len(self._treatment_name) > 1:
            raise NotImplementedError(
                "IvExclusionSensitivityRefuter does not currently support multiple endogenous treatments."
            )

        data = self._data
        treatment_name = self._treatment_name[0]
        outcome_name = self._outcome_name[0]
        instrument_names = self._target_estimand.instrumental_variables

        if not instrument_names:
            raise ValueError("No instrumental variables found in the causal model.")

        Z = data[instrument_names].values
        D = data[treatment_name].values
        Y = data[outcome_name].values

        # 1. Intercept Handling: Explicit check on the raw data matrix
        # Note: We apply this to raw data to ensure the projection matrix P_Z
        # accurately reflects models fitted with an intercept.
        has_intercept = any(np.allclose(Z[:, i], 1.0) for i in range(Z.shape[1]))
        if not has_intercept:
            Z = np.c_[np.ones(Z.shape[0]), Z]

        Z_T_Z_inv = np.linalg.pinv(Z.T @ Z)
        P_Z = Z @ Z_T_Z_inv @ Z.T

        D_mat = D.reshape(-1, 1) if D.ndim == 1 else D

        # 2. Conditioning Diagnostic
        # Check Z'Z for collinearity among instruments
        Z_T_Z = Z.T @ Z
        cond_number = np.linalg.cond(Z_T_Z)
        if cond_number > 1e4:
            warnings.warn(
                "This indicates numerical instability in the instrument matrix (Z'Z), "
                "which may arise from multicollinearity among instruments or poor scaling. "
                "Sensitivity estimates may be numerically unstable.",
                UserWarning,
            )

        D_P_Z_D = D_mat.T @ P_Z @ D_mat
        D_P_Z_D_inv = np.linalg.pinv(D_P_Z_D)
        A = D_P_Z_D_inv @ D_mat.T @ Z

        gamma_mean_vec = np.zeros(Z.shape[1])
        gamma_var_vec = np.zeros(Z.shape[1])

        if isinstance(self.gamma_prior_mean, dict) and isinstance(self.gamma_prior_var, dict):
            for i, inst in enumerate(instrument_names):
                offset = 0 if has_intercept else 1
                if inst in self.gamma_prior_mean:
                    gamma_mean_vec[i + offset] = self.gamma_prior_mean[inst]
                    gamma_var_vec[i + offset] = self.gamma_prior_var.get(inst, 0.0)
        else:
            if len(instrument_names) == 1:
                offset = 0 if has_intercept else 1
                gamma_mean_vec[offset] = self.gamma_prior_mean
                gamma_var_vec[offset] = self.gamma_prior_var
            else:
                raise ValueError(
                    "Multiple instruments detected. You must pass dictionaries for "
                    "`gamma_prior_mean` and `gamma_prior_var` mapping names to priors."
                )

        # 3. Variance Validation
        if np.any(gamma_var_vec < 0):
            raise ValueError("All gamma prior variances must be non-negative.")

        bias = A @ gamma_mean_vec
        bias_scalar = bias.item() if bias.size == 1 else bias[0]
        simulated_new_effect = self._estimate.value - bias_scalar

        Y_mat = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        beta_ltz_array = np.array([[simulated_new_effect]])
        residuals = Y_mat - (D_mat @ beta_ltz_array) - (Z @ gamma_mean_vec.reshape(-1, 1))

        # 4. Small-Sample Protection
        N = Z.shape[0]
        k = D_mat.shape[1] + Z.shape[1]
        if N <= k:
            raise ValueError(
                f"Degrees of freedom exhausted. Sample size (N={N}) must be strictly "
                f"greater than the number of parameters (k={k})."
            )

        sigma_sq = (residuals.T @ residuals) / (N - k)
        sigma_sq_scalar = sigma_sq.item() if sigma_sq.size == 1 else sigma_sq[0, 0]

        V_2sls = sigma_sq_scalar * D_P_Z_D_inv
        Omega_gamma = np.diag(gamma_var_vec)
        V_gamma = A @ Omega_gamma @ A.T

        V_LTZ = V_2sls + V_gamma

        # 5. PSD Protection for Floating Point Errors
        diag = np.maximum(np.diag(V_LTZ), 0)
        adjusted_se = np.sqrt(diag)[0]

        z_score = st.norm.ppf(0.975)
        ci_lower = simulated_new_effect - (z_score * adjusted_se)
        ci_upper = simulated_new_effect + (z_score * adjusted_se)

        refute = CausalRefutation(
            estimated_effect=self._estimate.value,
            new_effect=simulated_new_effect,
            refutation_type="IV Exclusion Sensitivity (Approximate Conley Analysis)",
        )

        refute.new_effect_standard_error = adjusted_se
        refute.new_effect_confidence_interval = (ci_lower, ci_upper)
        return refute
