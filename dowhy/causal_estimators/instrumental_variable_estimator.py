import numpy as np
import sympy as sp
import sympy.stats as spstats
from statsmodels.sandbox.regression.gmm import IV2SLS

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimator import RealizedEstimand
from dowhy.utils.api import parse_state


class InstrumentalVariableEstimator(CausalEstimator):
    """Compute effect of treatment using the instrumental variables method.

    This is also a superclass that can be inherited by other specific methods.


    Supports additional parameters that can be specified in the estimate_effect() method.

    - 'iv_instrument_name': Name of the specific instrumental variable to be used. Needs to be one of the IVs identified in the identification step. Default is to use all the IV variables from the identification step.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Instrumental Variables used:" +
                          ",".join(self._target_estimand.instrumental_variables))


        # choosing the instrumental variable to use
        if getattr(self, 'iv_instrument_name', None) is None:
            self.estimating_instrument_names = self._target_estimand.instrumental_variables
        else:
            self.estimating_instrument_names = parse_state(self.iv_instrument_name)

        if not self.estimating_instrument_names:
            raise ValueError("No valid instruments found. IV Method not applicable")
        if len(self.estimating_instrument_names) < len(self._treatment_name):
            # TODO move this to the identification step
            raise ValueError("Number of instruments fewer than number of treatments. 2SLS requires at least as many instruments as treatments.")
        self._estimating_instruments = self._data[self.estimating_instrument_names]
        self.logger.info("INFO: Using Instrumental Variable Estimator")

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        if len(self.estimating_instrument_names) == 1 and len(self._treatment_name) == 1:
            instrument = self._estimating_instruments.iloc[:,0]
            self.logger.debug("Instrument Variable values: {0}".format(instrument))
            num_unique_values = len(np.unique(instrument))
            instrument_is_binary = (num_unique_values <= 2)
            if instrument_is_binary:
                # Obtain estimate by Wald Estimator
                y1_z = np.mean(self._outcome[instrument == 1])
                y0_z = np.mean(self._outcome[instrument == 0])
                x1_z = np.mean(self._treatment[self._treatment_name[0]][instrument == 1])
                x0_z = np.mean(self._treatment[self._treatment_name[0]][instrument == 0])
                num = y1_z - y0_z
                deno = x1_z - x0_z
                iv_est = num / deno
            else:
                # Obtain estimate by 2SLS estimator: Cov(y,z) / Cov(x,z)
                num_yz = np.cov(self._outcome, instrument)[0, 1]
                deno_xz = np.cov(self._treatment[self._treatment_name[0]], instrument)[0, 1]
                iv_est = num_yz / deno_xz
        else:
            # More than 1 instrument. Use 2sls.
            est_treatment = self._treatment.astype(np.float32)
            est_outcome = self._outcome.astype(np.float32)
            ivmodel = IV2SLS(est_outcome, est_treatment,
                    self._estimating_instruments)
            reg_results = ivmodel.fit()
            print(reg_results.summary())
            iv_est = sum(reg_results.params) # the effect is the same for any treatment value (assume treatment goes from 0 to 1)
        estimate = CausalEstimate(estimate=iv_est,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        sym_outcome = (spstats.Normal(",".join(estimand.outcome_variable), 0, 1))
        sym_treatment = (spstats.Normal(",".join(estimand.treatment_variable), 0, 1))
        sym_instrument = sp.Symbol(",".join(self.estimating_instrument_names))
        sym_outcome_derivative = sp.Derivative(sym_outcome, sym_instrument)
        sym_treatment_derivative = sp.Derivative(sym_treatment, sym_instrument)
        sym_effect = (
                spstats.Expectation(sym_outcome_derivative) /
                sp.stats.Expectation(sym_treatment_derivative)
        )
        estimator_assumptions = {
            "treatment_effect_homogeneity": (
                "Each unit's treatment {0} is ".format(self._treatment_name) +
                "affected in the same way by common causes of "
                "{0} and {1}".format(self._treatment_name, self._outcome_name)
            ),
            "outcome_effect_homogeneity": (
                "Each unit's outcome {0} is ".format(self._outcome_name) +
                "affected in the same way by common causes of "
                "{0} and {1}".format(self._treatment_name, self._outcome_name)
            ),
        }
        sym_assumptions = {**estimand.estimands["iv"]["assumptions"],
                           **estimator_assumptions}

        symbolic_estimand = RealizedEstimand(estimand,
                                             estimator_name="Wald Estimator")
        symbolic_estimand.update_assumptions(sym_assumptions)
        symbolic_estimand.update_estimand_expression(sym_effect)
        return symbolic_estimand
