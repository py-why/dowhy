from .generalised_cov_measure import generalised_cov_based
from .kernel import approx_kernel_based, kernel_based
from .regression import regression_based


def independence_test(X, Y, conditioned_on=None, method="kernel", **kwargs):
    """Performs a (conditional) independence test.
    Three methods for (conditional) independence test are supported at the moment:

    * `kernel`: Kernel-based (conditional) independence test.

      * K. Zhang, J. Peters, D. Janzing, B. Schölkopf. *Kernel-based Conditional Independence Test and Application in Causal Discovery*. UAI'11, Pages 804–813, 2011.
      * A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, A. Smola. *A Kernel Statistical Test of Independence*. NIPS 21, 2007.
      Here, we utilize the implementations of the https://github.com/cmu-phil/causal-learn package.

    * `approx_kernel`: Approximate kernel-based (conditional) independence test.

      * E. Strobl, K. Zhang, S. Visweswaran. *Approximate kernel-based conditional independence tests for fast non-parametric causal discovery*. Journal of Causal Inference, 2019.

    * `regression`: Regression based (conditional) independence test using a f-test. See :func:`~dowhy.gcm.regression_based` for more details.

    * `gcm`: (Conditional) independence test based on the Generalised Covariance Measure. See :func:`~dowhy.gcm.generalised_cov_based` for more details.

        * R. D. Shah and J Peters. *The hardness of conditional independence testing and the generalised covariance measure*, The Annals of Statistics 48(3), 2018

    :param X: Observations of X.
    :param Y: Observations of Y.
    :param conditioned_on: Observations of conditioning variable if we want to perform a conditional independence test. By default, independence test is carried out.
    :param method: Method for conditional independence test. The choices are:
                   `kernel` (default): :func:`~dowhy.gcm.kernel_based` (conditional) independence test.
                   `approx_kernel`: :func:`~dowhy.gcm.approx_kernel_based` (conditional) independence test.
                   `regression`: :func:`~dowhy.gcm.regression_based` (conditional) independence test.
                   `gcm`: :func:`~dowhy.gcm.generalised_cov_based` (conditional) independence test.
                   For more information about these methods, see above.
    :return:  p-value of the (conditional) independence test. (Conditional) Independence is the null hypothesis.
    """
    if method == "kernel":
        return kernel_based(X, Y, Z=conditioned_on, **kwargs)
    elif method == "approx_kernel":
        return approx_kernel_based(X, Y, Z=conditioned_on, **kwargs)
    elif method == "regression":
        return regression_based(X, Y, Z=conditioned_on, **kwargs)
    elif method == "gcm":
        return generalised_cov_based(X, Y, Z=conditioned_on, **kwargs)
    else:
        raise ValueError(f'Invalid method "{method}"')
