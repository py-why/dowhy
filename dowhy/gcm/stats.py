from typing import Union, List, Optional

import numpy as np


def quantile_based_fwer(p_values: Union[np.ndarray, List[float]],
                        p_values_scaling: Optional[np.ndarray] = None,
                        quantile: float = 0.5) -> float:
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
