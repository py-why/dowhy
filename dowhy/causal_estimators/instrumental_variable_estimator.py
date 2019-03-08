import numpy as np
import sympy as sp
import sympy.stats as spstats

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimator import RealizedEstimand


class InstrumentalVariableEstimator(CausalEstimator):
    """Compute effect of treatment using the instrumental variables method.

    This is a superclass that is inherited by other specific methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Instrumental Variables used:" +
                          ",".join(self._target_estimand.instrumental_variables))
        self._instrument_names = self._target_estimand.instrumental_variables

        # choosing the instrumental variable to use
        if getattr(self, 'iv_instrument_name', None) is None:
            self._instruments = self._data[self._instrument_names]
            self.estimating_instrument = self._instruments[self._instrument_names[0]]
        else:
            self.estimating_instrument = self._data[self.iv_instrument_name]
        self.logger.info("INFO: Using Instrumental Variable Estimator")

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        instrument = self.estimating_instrument
        self.logger.debug("Instrument Variable values: {0}".format(instrument))
        num_unique_values = len(np.unique(instrument))
        instrument_is_binary = (num_unique_values <= 2)
        if instrument_is_binary:
            # Obtain estimate by Wald Estimator
            y1_z = np.mean(self._outcome[instrument == 1])
            y0_z = np.mean(self._outcome[instrument == 0])
            x1_z = np.mean(self._treatment[instrument == 1])
            x0_z = np.mean(self._treatment[instrument == 0])
            num = y1_z - y0_z
            deno = x1_z - x0_z
            iv_est = num / deno
        else:
            # Obtain estimate by Pearl (1995) ratio estimator.
            # y = x+ u; multiply both sides by z and take expectation.
            num_yz = np.dot(self._outcome, instrument)
            deno_xz = np.dot(self._treatment, instrument)
            iv_est = num_yz / deno_xz

        estimate = CausalEstimate(estimate=iv_est,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        sym_outcome = (spstats.Normal(",".join(estimand.outcome_variable), 0, 1))
        sym_treatment = (spstats.Normal(",".join(estimand.treatment_variable), 0, 1))
        sym_instrument = sp.Symbol(estimand.instrumental_variables[0])
        sym_outcome_derivative = sp.Derivative(sym_outcome, sym_instrument)
        sym_treatment_derivative = sp.Derivative(sym_treatment, sym_instrument)
        sym_effect = (
                spstats.Expectation(sym_outcome_derivative) /
                sp.stats.Expectation(sym_treatment_derivative)
        )
        estimator_assumptions = {
            "treatment_effect_homogeneity": (
                "Each unit's treatment {0} is".format(self._treatment_name) +
                "affected in the same way by common causes of "
                "{0} and {1}".format(self._treatment_name, self._outcome_name)
            ),
            "outcome_effect_homogeneity": (
                "Each unit's outcome {0} is".format(self._outcome_name) +
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
