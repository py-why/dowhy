"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Union, List, Optional

import numpy as np


def quantile_based_fwer(p_values: Union[np.ndarray, List[float]],
                        p_values_scaling: Optional[np.ndarray] = None,
                        quantile: float = 0.5) -> float:
    """Applies a quantile based family wise error rate (FWER) control to the given p-values. This is based on the
    approach described in:

    Meinshausen, N., Meier, L. and Buehlmann, P. (2009).
    p-values for high-dimensional regression. J. Amer. Statist. Assoc.104 1671–1681

    :param p_values: A list or array of p-values.
    :param p_values_scaling: An optional list of scaling factors for each p-value.
    :param quantile: The quantile used for the p-value adjustment. By default, this is the median (0.5).
    :return: The p-value that lies on the quantile threshold. Note that this is the quantile based on scaled values
             p_values / quantile.
    """

    if quantile <= 0 or abs(quantile - 1) >= 1:
        raise ValueError("The given quantile is %f, but it needs to be on (0, 1]!" % quantile)

    p_values = np.array(p_values)
    if p_values_scaling is None:
        p_values_scaling = np.ones(p_values.shape[0])

    if p_values.shape != p_values_scaling.shape:
        raise ValueError("The p-value scaling array needs to have the same dimension as the given p-values.")

    p_values_scaling = p_values_scaling[~np.isnan(p_values)]
    p_values = p_values[~np.isnan(p_values)]

    p_values = p_values * p_values_scaling
    p_values[p_values > 1] = 1.0

    if p_values.shape[0] == 1:
        return float(p_values[0])
    else:
        return float(min(1.0, np.quantile(p_values / quantile, quantile)))
