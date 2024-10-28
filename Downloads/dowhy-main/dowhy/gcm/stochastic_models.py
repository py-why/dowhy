"""This module defines multiple implementations of the abstract class :class:`~dowhy.gcm.graph.StochasticModel`."""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy
from scipy.stats import norm, rv_continuous, rv_discrete
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture

from dowhy.gcm.causal_mechanisms import StochasticModel
from dowhy.gcm.util.general import shape_into_2d

_CONTINUOUS_DISTRIBUTIONS = [
    scipy.stats.norm,
    scipy.stats.laplace,
    scipy.stats.t,
    scipy.stats.uniform,
    scipy.stats.rayleigh,
]
_CONTINUOUS_DISTRIBUTIONS.extend(
    [
        getattr(scipy.stats, d)
        for d in dir(scipy.stats)
        if isinstance(getattr(scipy.stats, d), scipy.stats.rv_continuous) and d not in _CONTINUOUS_DISTRIBUTIONS
    ]
)

_DISCRETE_DISTRIBUTIONS = [
    getattr(scipy.stats, d) for d in dir(scipy.stats) if isinstance(getattr(scipy.stats, d), scipy.stats.rv_discrete)
]

_CONTINUOUS_DISTRIBUTIONS = {x.name: x for x in _CONTINUOUS_DISTRIBUTIONS}
_DISCRETE_DISTRIBUTIONS = {x.name: x for x in _DISCRETE_DISTRIBUTIONS}


class ScipyDistribution(StochasticModel):
    """Represents any parametric distribution that can be modeled by scipy."""

    def __init__(self, scipy_distribution: Optional[Union[rv_continuous, rv_discrete]] = None, **parameters) -> None:
        """Initializes a stochastic model that allows to sample from a parametric distribution implemented in Scipy.

        For instance, to use a beta distribution with parameters a=2 and b=0.5:
            ScipyDistribution(stats.beta, a=2, b=0.5)
        Or a Gaussian distribution with mean=0 and standard deviation 2:
            ScipyDistribution(stats.norm, loc=2, scale=0.5)

        Note that the parameter names need to coincide with the parameter names in the corresponding Scipy
        implementations. See https://docs.scipy.org/doc/scipy/tutorial/stats.html for more information.

        :param scipy_distribution: A continuous or discrete distribution parametric distribution implemented in Scipy.
        :param parameters: Set of parameters of the parametric distribution.
        """
        self._distribution = scipy_distribution
        self._parameters = parameters
        self._fixed_parameters = len(parameters) > 0

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if len(self._parameters) == 0 or self._distribution is None:
            raise ValueError("Cannot draw samples. Model has not been fit!")

        return shape_into_2d(self._distribution.rvs(size=num_samples, **self.parameters))

    def fit(self, X: np.ndarray) -> None:
        if self._distribution is None:
            # Currently only support continuous distributions for auto selection.
            best_model, best_parameters = self.find_suitable_continuous_distribution(X)
            self._distribution = best_model
            self._parameters = best_parameters
        elif not self._fixed_parameters:
            self._parameters = self.map_scipy_distribution_parameters_to_names(
                self._distribution, self._distribution.fit(shape_into_2d(X))
            )

    @property
    def parameters(self) -> Dict[str, float]:
        return self._parameters

    @property
    def scipy_distribution(self) -> Optional[Union[rv_continuous, rv_discrete]]:
        return self._distribution

    def clone(self):
        if self._fixed_parameters:
            return ScipyDistribution(scipy_distribution=self._distribution, **self._parameters)
        else:
            return ScipyDistribution(scipy_distribution=self._distribution)

    @staticmethod
    def find_suitable_continuous_distribution(
        distribution_samples: np.ndarray, divergence_threshold: float = 10**-2
    ) -> Tuple[rv_continuous, Dict[str, float]]:
        """Tries to find the best fitting continuous parametric distribution of given samples. This is done by fitting
        different parametric models and selecting the one with the smallest KL divergence between observed and generated
        samples.
        """
        distribution_samples = shape_into_2d(distribution_samples)

        currently_best_distribution = norm
        currently_best_parameters = (0.0, 1.0)
        currently_smallest_divergence = np.inf

        # Estimate distribution parameters from data.
        for distribution in _CONTINUOUS_DISTRIBUTIONS.values():
            # Ignore warnings from fitting process.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                try:
                    # Fit distribution to data.
                    params = distribution.fit(distribution_samples)
                except ValueError:
                    # Some distributions might not be compatible with the data.
                    continue

                # Separate parts of parameters.
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                generated_samples = distribution.rvs(size=distribution_samples.shape[0], loc=loc, scale=scale, *arg)

                # Check the KL divergence between the distribution of the given and fitted distribution.
                from dowhy.gcm.divergence import estimate_kl_divergence_continuous_knn

                divergence = estimate_kl_divergence_continuous_knn(distribution_samples, generated_samples)
                if divergence < divergence_threshold:
                    currently_best_distribution = distribution
                    currently_best_parameters = params
                    break

                # Identify if this distribution is better.
                if currently_smallest_divergence > divergence:
                    currently_best_distribution = distribution
                    currently_best_parameters = params
                    currently_smallest_divergence = divergence

        return currently_best_distribution, ScipyDistribution.map_scipy_distribution_parameters_to_names(
            currently_best_distribution, currently_best_parameters
        )

    @staticmethod
    def map_scipy_distribution_parameters_to_names(
        scipy_distribution: Union[rv_continuous, rv_discrete], parameters: Tuple[float]
    ) -> Dict[str, float]:
        """Helper function to obtain a mapping from parameter name to parameter value. Depending whether the
        distribution is discrete or continuous, there are slightly different parameter names. The given parameters are
        assumed to follow the order as provided by the scipy fit function.

        :param scipy_distribution: The scipy distribution.
        :param parameters: The values of the corresponding parameters of the distribution. Here, it is expected to
                           follow the same order as defined by the scipy fit function.
        :return: A dictionary that maps a parameter name to its value.
        """
        if scipy_distribution.shapes:
            parameter_list = [name.strip() for name in scipy_distribution.shapes.split(",")]
        else:
            parameter_list = []
        if scipy_distribution.name in _DISCRETE_DISTRIBUTIONS:
            parameter_list += ["loc"]
        elif scipy_distribution.name in _CONTINUOUS_DISTRIBUTIONS:
            parameter_list += ["loc", "scale"]
        else:
            raise ValueError(
                "Distribution %s not found in the list of continuous and discrete distributions!"
                % scipy_distribution.name
            )

        parameters_dictionary = {}
        for i, parameter_name in enumerate(parameter_list):
            parameters_dictionary[parameter_name] = parameters[i]

        return parameters_dictionary

    def __str__(self) -> str:
        return str(self._distribution.name) + " distribution"


class EmpiricalDistribution(StochasticModel):
    """An implementation of a stochastic model that uniformly samples from data samples. By randomly returning a sample
    from the training data set, this model represents a parameter free representation of the marginal distribution of
    the training data. However, it will not generate unseen data points. For this, consider :py:class:`BayesianGaussianMixtureDistribution <dowhy.gcm.BayesianGaussianMixtureDistribution>`.
    """

    def __init__(self) -> None:
        self._data = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    def fit(self, X: np.ndarray) -> None:
        self._data = shape_into_2d(X)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("%s has not been fitted!" % self.__class__.__name__)

        return self.data[np.random.choice(self.data.shape[0], size=num_samples, replace=True), :]

    def clone(self):
        return EmpiricalDistribution()

    def __str__(self):
        return "Empirical Distribution"


class BayesianGaussianMixtureDistribution(StochasticModel):
    def __init__(self) -> None:
        self._gmm_model = None

    def fit(self, X: np.ndarray) -> None:
        X = shape_into_2d(X)
        self._gmm_model = BayesianGaussianMixture(
            n_components=BayesianGaussianMixtureDistribution._get_optimal_number_of_components(X), max_iter=1000
        ).fit(X)

    @staticmethod
    def _get_optimal_number_of_components(X: np.ndarray) -> int:
        current_best = 0
        current_best_num_components = 1
        num_best_in_succession = 0
        try:
            for i in range(2, int(np.sqrt(X.shape[0] / 2))):
                kmeans = KMeans(n_clusters=i).fit(X)
                coefficient = silhouette_score(X, kmeans.labels_, sample_size=5000)

                if coefficient > current_best:
                    current_best = coefficient
                    current_best_num_components = i
                    num_best_in_succession = 0
                else:
                    num_best_in_succession += 1

                if num_best_in_succession >= 3:
                    break
        except ValueError:
            # This error is typically raised when the data is discrete and all points are assigned to less cluster than
            # specified. It can also happen due to duplicated points. In these cases, the current best solution should
            # be sufficient.
            return current_best_num_components

        return current_best_num_components

    def draw_samples(self, num_samples: int) -> np.ndarray:
        if self._gmm_model is None:
            raise RuntimeError("%s has not been fitted!" % self.__class__.__name__)

        return shape_into_2d(self._gmm_model.sample(num_samples)[0])

    def __str__(self) -> str:
        return "Gaussian Mixture Distribution"

    def clone(self):
        return BayesianGaussianMixtureDistribution()
