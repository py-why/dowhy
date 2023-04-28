"""This module is deprecated! All functions are moved into the feature_relevance.py module. """
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dowhy.gcm import feature_relevance
from dowhy.gcm.causal_models import StructuralCausalModel
from dowhy.gcm.shapley import ShapleyConfig


def parent_relevance(
    causal_model: StructuralCausalModel,
    target_node: Any,
    parent_samples: Optional[pd.DataFrame] = None,
    subset_scoring_func: Optional[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]] = None,
    num_samples_randomization: int = 5000,
    num_samples_baseline: int = 500,
    max_batch_size: int = 100,
    shapley_config: Optional[ShapleyConfig] = None,
) -> Tuple[Dict[Any, Any], np.ndarray]:
    """Deprecated, please use parent_relevance from the feature_relevance.py module instead."""
    warnings.warn(
        "This module is deprecated. All feature.py functions are moved into parent_relevance.py.", DeprecationWarning
    )
    return feature_relevance.parent_relevance(
        causal_model=causal_model,
        target_node=target_node,
        parent_samples=parent_samples,
        subset_scoring_func=subset_scoring_func,
        num_samples_randomization=num_samples_randomization,
        num_samples_baseline=num_samples_baseline,
        max_batch_size=max_batch_size,
        shapley_config=shapley_config,
    )


def feature_relevance_distribution(
    prediction_method: Callable[[np.ndarray], np.ndarray],
    feature_samples: np.ndarray,
    subset_scoring_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    max_num_samples_randomization: int = 5000,
    max_num_baseline_samples: int = 500,
    max_batch_size: int = 100,
    randomize_features_jointly: bool = True,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Deprecated, please use feature_relevance_distribution from the feature_relevance.py module instead."""
    warnings.warn(
        "This module is deprecated. All feature.py functions are moved into parent_relevance.py.", DeprecationWarning
    )
    return feature_relevance.feature_relevance_distribution(
        prediction_method=prediction_method,
        feature_samples=feature_samples,
        subset_scoring_func=subset_scoring_func,
        max_num_samples_randomization=max_num_samples_randomization,
        max_num_baseline_samples=max_num_baseline_samples,
        max_batch_size=max_batch_size,
        randomize_features_jointly=randomize_features_jointly,
        shapley_config=shapley_config,
    )


def feature_relevance_sample(
    prediction_method: Callable[[np.ndarray], np.ndarray],
    feature_samples: np.ndarray,
    baseline_samples: np.ndarray,
    subset_scoring_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    baseline_target_values: Optional[np.ndarray] = None,
    average_set_function: bool = False,
    max_batch_size: int = 100,
    randomize_features_jointly: bool = True,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Deprecated, please use feature_relevance_sample from the feature_relevance.py module instead."""
    warnings.warn(
        "This module is deprecated. All feature.py functions are moved into parent_relevance.py.", DeprecationWarning
    )
    return feature_relevance.feature_relevance_sample(
        prediction_method=prediction_method,
        feature_samples=feature_samples,
        baseline_samples=baseline_samples,
        subset_scoring_func=subset_scoring_func,
        baseline_target_values=baseline_target_values,
        average_set_function=average_set_function,
        max_batch_size=max_batch_size,
        randomize_features_jointly=randomize_features_jointly,
        shapley_config=shapley_config,
    )
