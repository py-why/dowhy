import numpy as np
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kernel_density import EstimatorSettings, KDEMultivariateConditional

from dowhy.do_sampler import DoSampler


class KernelDensitySampler(DoSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info("Using KernelDensitySampler for do sampling.")
        if (
            len(self._data) > 300
            or max(
                len(self._treatment_names + self._target_estimand.get_adjustment_set()),
                len(self._outcome_names + self._target_estimand.get_adjustment_set()),
            )
            >= 3
        ):
            self.defaults = EstimatorSettings(n_jobs=4, efficient=True)
        else:
            self.defaults = EstimatorSettings(n_jobs=-1, efficient=False)

        if "c" not in self._variable_types.values():
            self.bw = "cv_ml"
        else:
            self.bw = "normal_reference"

        self.sampler = self._construct_sampler()

    def _fit_conditional(self):
        self.conditional_density = KDEMultivariateConditional(
            endog=self._data[self._outcome_names],
            exog=self._data[self._treatment_names + self._target_estimand.get_adjustment_set()],
            dep_type="".join(self.dep_type),
            indep_type="".join(self.indep_type),
            bw=self.bw,
            defaults=self.defaults,
        )

    def _infer_variable_types(self):
        raise Exception(
            "Variable type inference not implemented. Specify variable_types={var_name: var_type}, "
            "where var_type is 'o', 'c', or 'd' for ordered, continuous, or discrete, respectively."
        )

    def _construct_sampler(self):
        return KernelSampler(
            self.outcome_upper_support,
            self.outcome_lower_support,
            self._outcome_names,
            self._treatment_names,
            self._target_estimand.get_adjustment_set(),
            self._data,
            self.dep_type,
            self.indep_type,
            self.bw,
            self.defaults,
        )


class KernelSampler(object):
    def __init__(
        self,
        outcome_upper_support,
        outcome_lower_support,
        outcome_names,
        treatment_names,
        backdoor_variables,
        data,
        dep_type,
        indep_type,
        bw,
        defaults,
    ):
        self._data = data
        self._outcome_names = outcome_names
        self._treatment_names = treatment_names
        self._backdoor_variables = backdoor_variables
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.bw = bw
        self.defaults = defaults
        self.outcome_lower_support = outcome_lower_support
        self.outcome_upper_support = outcome_upper_support
        self.conditional_density = KDEMultivariateConditional(
            endog=self._data[self._outcome_names],
            exog=self._data[self._treatment_names + self._backdoor_variables],
            dep_type="".join(self.dep_type),
            indep_type="".join(self.indep_type),
            bw=self.bw,
            defaults=self.defaults,
        )

    def sample_point(self, x_z):
        y_bw = 1.06 * self._data[self._outcome_names].std() * (self._data[self._outcome_names].count()) ** (-1.0 / 5.0)
        n = 5 * np.ceil((self.outcome_upper_support - self.outcome_lower_support) / y_bw)
        cum_ranges = [
            np.linspace(self.outcome_lower_support[i], self.outcome_upper_support[i], n[i])
            for i in range(len(self._outcome_names))
        ]

        res = np.meshgrid(*cum_ranges)
        points = np.array(res).reshape(len(self._outcome_names), np.int(n.cumprod()[-1])).T

        x_z_repeated = np.repeat(x_z, len(points)).reshape(len(points), len(x_z))
        cdf_vals = self._evaluate_cdf(points, x_z_repeated)
        cdf_vals = np.hstack([[0.0], cdf_vals, [1.0]])
        points = np.vstack(
            [[self.outcome_lower_support - 3.0 * y_bw], points, [self.outcome_upper_support + 3.0 * y_bw]]
        )
        inv_cdf = interp1d(cdf_vals.flatten(), points.flatten(), fill_value=0.0, axis=0)
        r = np.random.rand()
        try:
            return inv_cdf(r)
        except ValueError:
            return self.sample_point(x_z)

    def _evaluate_cdf(self, y, x_z):
        return self.conditional_density.cdf(endog_predict=[y], exog_predict=x_z)
