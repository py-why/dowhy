from dowhy.do_sampler import DoSampler
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional, KDEMultivariate, EstimatorSettings
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator


class MCMCSampler(DoSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info("Using MCMCSampler for do sampling.")
        self.sampler = self._construct_sampler()

    def _construct_sampler(self):
