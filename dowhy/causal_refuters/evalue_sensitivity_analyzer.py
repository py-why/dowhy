import logging
import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.linear_regression_estimator import (
    LinearRegressionEstimator,
)
from dowhy.causal_estimators.causalml import Causalml
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_estimators.instrumental_variable_estimator import (
    InstrumentalVariableEstimator,
)
from dowhy.causal_estimators.regression_discontinuity_estimator import (
    RegressionDiscontinuityEstimator,
)
from dowhy.causal_estimators.two_stage_regression_estimator import (
    TwoStageRegressionEstimator,
)


class EValueSensitivityAnalyzer:
    """
    This class computes Ding & VanderWeele's E-value for unmeasured confounding. The E-value is the minimum
    strength of association on the risk ratio scale that an unmeasured confounder would need to have with
    both the treatment and the outcome, conditional on the measured covariates, to fully explain away a specific
    treatment-outcome association.

    See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4820664/ and
    https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf.
    The implementation is based on the R package: https://github.com/cran/EValue.

    :param estimate: CausalEstimate
    :param outcome_var: Outcome variable values
    :param true: A number to which to shift the observed estimate to. Defaults to 1 for ratio measures (RR, OR, HR)
    and 0 for additive measures (OLS, MD). (Default = 0)
    :param delta: The contrast of interest in the treatment/exposure, for estimator of class LinearRegressionEstimator.
    (Default = 1)
    """

    def __init__(
        self, estimate: CausalEstimate, outcome_var: pd.Series, true=0, delta=1
    ):

        self.estimate = estimate
        self.true = true
        self.delta = delta
        self.logger = logging.getLogger(__name__)
        self.sd_outcome = np.std(outcome_var)
        self.stats = None

        outcome_values = outcome_var.astype(int).unique()
        if all([v in [0, 1] for v in outcome_values]):
            raise NotImplementedError(
                "E-Value sensitivity analysis is not currently implemented for binary outcomes"
            )
        if isinstance(
            self.estimate.estimator,
            (
                Causalml,
                Econml,
                InstrumentalVariableEstimator,
                RegressionDiscontinuityEstimator,
                TwoStageRegressionEstimator,
            ),
        ):
            raise NotImplementedError(
                "E-Value sensitivity analysis is not currently implemented for this estimator"
            )

    def check_sensitivity(self):
        """
        Function to compute E-value for point estimate and confidence limits. The estimate and
        confidence limits are converted to the risk ratio scale before the E-value is calculated.
        """

        est = self.estimate.value
        se = self.estimate.get_standard_error()
        if isinstance(se, np.ndarray):
            se = se[0]

        if isinstance(self.estimate.estimator, LinearRegressionEstimator):
            self._evalue_OLS(est, se, self.sd_outcome, self.delta, self.true)
        else:
            est = self._to_md(est, self.sd_outcome)
            se = self._to_md(se, self.sd_outcome)
            self._evalue_MD(est, se, self.true)

    def _evalue_OLS(self, est, se, sd, delta=1, true=0):
        """
        Function to compute E-value from OLS coefficient.

        :param est: coefficient from OLS
        :param se: standard error of point estimate
        :param sd: residual standard deviation
        :param delta: contrast of interest in the treatment/exposure
        :param true: true standardized difference to which to shift the observed estimate. (Default = 0)
        """
        if est < 0:
            raise ValueError("Standard error cannot be negative")

        if delta < 0:
            delta = -delta
            self.logger.info("Recoding delta to be positive")

        est = self._to_md(est, sd, delta=delta)
        se = self._to_md(se, sd, delta=delta)
        print(f"est {est}, se {se}")
        self._evalue_MD(est=est, se=se, true=true)

    def _evalue_MD(self, est, se=None, true=0):
        """
        Function to compute E-value for standardized difference and its confidence limits.

        :param est: point estimate as standardized difference (i.e., Cohen's d)
        :param se: standard error of the point estimate
        :param true: true standardized difference to which to shift the observed estimate. (Default = 0)
        """
        if est < 0:
            raise ValueError("Standard error cannot be negative")

        if se is None:
            lo = None
            hi = None
        else:
            lo = np.exp(0.91 * est - 1.78 * se)
            hi = np.exp(0.91 * est + 1.78 * se)

        est = self._md_to_rr(est)
        true = self._md_to_rr(true)

        self._evalue_RR(est=est, lo=lo, hi=hi, true=true)

    def _evalue_OR(self, est, lo=None, hi=None, rare=None, true=1):
        """
        Function to compute E-value for an odds ratio and its confidence limits.

        :param est: point estimate
        :param lo: lower limit of confidence interval
        :param hi: upper limit of confidence interval
        :param rare: if outcome is rare (<15%)
        :param true: the true OR to which to shift the observed estimate. (Default = 1)
        """
        if est < 0:
            raise ValueError("Odds Ratio cannot be negative")

        est = self._or_to_rr(est, rare)
        if lo is not None:
            lo = self._or_to_rr(lo, rare)
        if hi is not None:
            hi = self._or_to_rr(hi, rare)
        true = self._or_to_rr(true, rare)

        self._evalue_RR(est=est, lo=lo, hi=hi, true=true)

    def _evalue_HR(self, est, lo=None, hi=None, rare=None, true=1):
        """
        Function to compute E-value for a hazard ratio and its confidence limits.

        :param est: point estimate
        :param lo: lower limit of confidence interval
        :param hi: upper limit of confidence interval
        :param rare: if outcome is rare (<15%)
        :param true: the true HR to which to shift the observed estimate. (Default = 1)
        """
        if est < 0:
            raise ValueError("Hazard Ratio cannot be negative")

        est = self._hr_to_rr(est, rare)
        if lo is not None:
            lo = self._hr_to_rr(lo, rare)
        if hi is not None:
            hi = self._hr_to_rr(hi, rare)
        true = self._hr_to_rr(true, rare)

        self._evalue_RR(est=est, lo=lo, hi=hi, true=true)

    def _evalue_RR(self, est, lo=None, hi=None, true=1):
        """
        Function to compute E-value for a risk ratio or rate ratio and its confidence limits.

        :param est: point estimate
        :param lo: lower limit of confidence interval
        :param hi: upper limit of confidence interval
        :param true: the true RR to which to shift the observed estimate. (Default = 1)
        """
        if est < 0:
            raise ValueError("Risk/Rate Ratio cannot be negative")
        if true < 0:
            raise ValueError("True value is impossible")
        if true != 1:
            self.logger.info(
                'You are calculating a "non-null" E-value, i.e., an E-value for the minimum amount of unmeasured '
                'confounding needed to move the estimate and confidence interval to your specified true value '
                'rather than to the null value.'
            )
        if lo is not None and hi is not None:
            if lo > hi:
                raise ValueError(
                    "Lower confidence limit should be less than upper confidence limit"
                )
        if lo is not None and est < lo:
            raise ValueError("Point estimate should be inside confidence interval")
        if hi is not None and est > hi:
            raise ValueError("Point estimate should be inside confidence interval")

        e_est = self._threshold(est, true=true)
        e_lo = self._threshold(lo, true=true)
        e_hi = self._threshold(hi, true=true)

        # if CI crosses null, set its E-value to 1
        null_CI = False
        if est > true and lo is not None:
            null_CI = lo < true
        if est < true and hi is not None:
            null_CI = hi > true
        if null_CI:
            e_lo = np.float64(1)
            e_hi = np.float64(1)

        # only report E-value for CI limit closer to null
        if lo is not None or hi is not None:
            if est > true:
                e_hi = None
            else:
                e_lo = None

        self.stats = {
            "converted_estimate": est,
            "converted_lower_ci": lo,
            "converted_upper_ci": hi,
            "evalue_estimate": e_est,
            "evalue_lower_ci": e_lo,
            "evalue_upper_ci": e_hi,
        }

    def _threshold(self, x, true=1):
        """
        Function to compute E-value for single value of risk ratio.

        :param x: risk ratio
        :param true: the true RR to which to shit the observed estimate
        """
        if x is None:
            return None

        if x <= 1:
            x = 1 / x
            true = 1 / true

        if true <= x:
            return (x + np.sqrt(x * (x - true))) / true
        else:
            ratio = true / x
            return ratio + np.sqrt(ratio * (ratio - 1))

    def _to_md(self, est, sd, delta=1):
        """
        Function to convert estimate to standardized mean difference.

        :param est: estimate
        :param sd: residual standard deviation or standard deviation of outcome
        :param delta: contrast of interest in the treatment/exposure
        """
        return est * delta / sd

    def _md_to_rr(self, est):
        """
        Function to convert standardized mean difference to approximate risk ratio.

        :param est: estimate
        """
        return np.exp(0.91 * est)

    def _hr_to_rr(self, est, rare):
        """
        Function to convert hazard ratio to approximate risk ratio.

        :param est: estimate
        :param rare: if outcome is rare (<15%)
        """
        if rare is None:
            raise ValueError(
                'Must specify whether the rare outcome assumption can be made. Use argument "rare" ='
            )
        if rare:
            return est
        else:
            return (1 - 0.5 ** np.sqrt(est)) / ((1 - 0.5 ** np.sqrt(1 / est)))

    def _or_to_rr(self, est, rare):
        """
        Function to convert odds ratio to approximate risk ratio.

        :param est: estimate
        :param rare: if outcome is rare (<15%)
        """
        if rare:
            return est
        else:
            return np.sqrt(est)
