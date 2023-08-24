import warnings
from enum import Enum, auto
from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from dowhy.gcm import config
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, ClassifierFCM
from dowhy.gcm.causal_models import CAUSAL_MECHANISM, ProbabilisticCausalModel, validate_causal_model_assignment
from dowhy.gcm.ml import (
    ClassificationModel,
    PredictionModel,
    create_hist_gradient_boost_classifier,
    create_hist_gradient_boost_regressor,
    create_lasso_regressor,
    create_linear_regressor,
    create_logistic_regression_classifier,
    create_random_forest_regressor,
    create_ridge_regressor,
    create_support_vector_regressor,
)
from dowhy.gcm.ml.classification import (
    create_ada_boost_classifier,
    create_extra_trees_classifier,
    create_gaussian_nb_classifier,
    create_knn_classifier,
    create_polynom_logistic_regression_classifier,
    create_random_forest_classifier,
    create_support_vector_classifier,
)
from dowhy.gcm.ml.regression import (
    create_ada_boost_regressor,
    create_extra_trees_regressor,
    create_knn_regressor,
    create_polynom_regressor,
)
from dowhy.gcm.stochastic_models import EmpiricalDistribution
from dowhy.gcm.util.general import (
    auto_apply_encoders,
    auto_fit_encoders,
    is_categorical,
    set_random_seed,
    shape_into_2d,
)
from dowhy.graph import get_ordered_predecessors, is_root_node

_LIST_OF_POTENTIAL_CLASSIFIERS_GOOD = [
    partial(create_logistic_regression_classifier, max_iter=1000),
    create_hist_gradient_boost_classifier,
]
_LIST_OF_POTENTIAL_REGRESSORS_GOOD = [
    create_linear_regressor,
    create_hist_gradient_boost_regressor,
]

_LIST_OF_POTENTIAL_CLASSIFIERS_BETTER = _LIST_OF_POTENTIAL_CLASSIFIERS_GOOD + [
    create_random_forest_classifier,
    create_extra_trees_classifier,
    create_support_vector_classifier,
    create_knn_classifier,
    create_gaussian_nb_classifier,
    create_ada_boost_classifier,
]
_LIST_OF_POTENTIAL_REGRESSORS_BETTER = _LIST_OF_POTENTIAL_REGRESSORS_GOOD + [
    create_ridge_regressor,
    create_polynom_regressor,
    partial(create_lasso_regressor, max_iter=5000),
    create_random_forest_regressor,
    create_support_vector_regressor,
    create_extra_trees_regressor,
    create_knn_regressor,
    create_ada_boost_regressor,
]


class AssignmentQuality(Enum):
    GOOD = auto()
    BETTER = auto()
    BEST = auto()


def assign_causal_mechanisms(
    causal_model: ProbabilisticCausalModel,
    based_on: pd.DataFrame,
    quality: AssignmentQuality = AssignmentQuality.GOOD,
    override_models: bool = False,
) -> None:
    """Automatically assigns appropriate causal models. If causal models are already assigned to nodes and
    override_models is set to False, this function only validates the assignments with respect to the graph structure.
    Here, the validation checks whether root nodes have StochasticModels and non-root ConditionalStochasticModels
    assigned.

    :param causal_model: The causal model to whose nodes to assign causal models.
    :param based_on: Jointly sampled data corresponding to the nodes of the given graph.
    :param quality: AssignmentQuality for the automatic model selection and model accuracy. This changes the type of
    prediction model and time spent on the selection. Options are:
        - AssignmentQuality.GOOD: Compares a linear, polynomial and gradient boost model on small test-training split
            of the data. The best performing model is then selected.
            Model selection speed: Fast
            Model training speed: Fast
            Model inference speed: Fast
            Model accuracy: Medium
        - AssignmentQuality.BETTER: Compares multiple model types and uses the one with the best performance
            averaged over multiple splits of the training data. By default, the model with the smallest root mean
            squared error is selected for regression problems and the model with the highest F1 score is selected for
            classification problems. For a list of possible models, see _LIST_OF_POTENTIAL_REGRESSORS_BETTER and
            _LIST_OF_POTENTIAL_CLASSIFIERS_BETTER, respectively.
            Model selection speed: Medium
            Model training speed: Fast
            Model inference speed: Fast
            Model accuracy: Good
        - AssignmentQuality.BEST: Uses an AutoGluon (auto ML) model with default settings defined by the AutoGluon
            wrapper. While the model selection itself is fast, the training and inference speed can be significantly
            slower than in the other options. NOTE: This requires the optional autogluon.tabular dependency.
            Model selection speed: Instant
            Model training speed: Slow
            Model inference speed: Slow-Medium
            Model accuracy: Best
        :param override_models: If set to True, existing model assignments are replaced with automatically selected
        ones. If set to False, the assigned models are only validated with respect to the graph structure.

    :return: None
    """
    for node in causal_model.graph.nodes:
        if not override_models and CAUSAL_MECHANISM in causal_model.graph.nodes[node]:
            validate_causal_model_assignment(causal_model.graph, node)
            continue
        assign_causal_mechanism_node(causal_model, node, based_on, quality)


def assign_causal_mechanism_node(
    causal_model: ProbabilisticCausalModel,
    node: str,
    based_on: pd.DataFrame,
    quality: AssignmentQuality = AssignmentQuality.GOOD,
) -> None:
    if is_root_node(causal_model.graph, node):
        causal_model.set_causal_mechanism(node, EmpiricalDistribution())
    else:
        prediction_model = select_model(
            based_on[get_ordered_predecessors(causal_model.graph, node)].to_numpy(),
            based_on[node].to_numpy(),
            quality,
        )

        if isinstance(prediction_model, ClassificationModel):
            causal_model.set_causal_mechanism(node, ClassifierFCM(prediction_model))
        else:
            causal_model.set_causal_mechanism(node, AdditiveNoiseModel(prediction_model))


def select_model(
    X: np.ndarray, Y: np.ndarray, model_selection_quality: AssignmentQuality
) -> Union[PredictionModel, ClassificationModel]:
    if model_selection_quality == AssignmentQuality.BEST:
        try:
            from dowhy.gcm.ml.autogluon import AutoGluonClassifier, AutoGluonRegressor

            if is_categorical(Y):
                return AutoGluonClassifier()
            else:
                return AutoGluonRegressor()
        except ImportError:
            raise RuntimeError(
                "AutoGluon module not found! For the BEST auto assign quality, consider installing the "
                "optional AutoGluon dependency."
            )
    elif model_selection_quality == AssignmentQuality.GOOD:
        list_of_regressor = list(_LIST_OF_POTENTIAL_REGRESSORS_GOOD)
        list_of_classifier = list(_LIST_OF_POTENTIAL_CLASSIFIERS_GOOD)
        model_selection_splits = 2
    elif model_selection_quality == AssignmentQuality.BETTER:
        list_of_regressor = list(_LIST_OF_POTENTIAL_REGRESSORS_BETTER)
        list_of_classifier = list(_LIST_OF_POTENTIAL_CLASSIFIERS_BETTER)
        model_selection_splits = 5
    else:
        raise ValueError("Invalid model selection quality.")

    if auto_apply_encoders(X, auto_fit_encoders(X)).shape[1] <= 5:
        # Avoid too many features
        list_of_regressor += [create_polynom_regressor]
        list_of_classifier += [partial(create_polynom_logistic_regression_classifier, max_iter=1000)]

    if is_categorical(Y):
        return find_best_model(list_of_classifier, X, Y, model_selection_splits=model_selection_splits)()
    else:
        return find_best_model(list_of_regressor, X, Y, model_selection_splits=model_selection_splits)()


def has_linear_relationship(X: np.ndarray, Y: np.ndarray, max_num_samples: int = 3000) -> bool:
    X, Y = shape_into_2d(X, Y)

    target_is_categorical = is_categorical(Y)
    # Making sure there are at least 30% test samples.
    num_trainings_samples = min(max_num_samples, round(X.shape[0] * 0.7))
    num_test_samples = min(X.shape[0] - num_trainings_samples, max_num_samples)

    if target_is_categorical:
        all_classes, indices, counts = np.unique(Y, return_counts=True, return_index=True)
        for i in range(all_classes.size):
            # Making sure that there are at least 2 samples from one class (here, simply duplicate the point).
            if counts[i] == 1:
                X = np.row_stack([X, X[indices[i], :]])
                Y = np.row_stack([Y, Y[indices[i], :]])

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, train_size=num_trainings_samples, test_size=num_test_samples, stratify=Y
        )

    else:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, train_size=num_trainings_samples, test_size=num_test_samples
        )

    encoders = auto_fit_encoders(x_train, y_train)
    x_train = auto_apply_encoders(x_train, encoders)
    x_test = auto_apply_encoders(x_test, encoders)

    if target_is_categorical:
        linear_mdl = LogisticRegression(max_iter=1000)
        nonlinear_mdl = create_hist_gradient_boost_classifier()
        linear_mdl.fit(x_train, y_train.squeeze())
        nonlinear_mdl.fit(x_train, y_train.squeeze())

        # Compare number of correct classifications.
        return np.sum(shape_into_2d(linear_mdl.predict(x_test)) == y_test) >= np.sum(
            shape_into_2d(nonlinear_mdl.predict(x_test)) == y_test
        )
    else:
        linear_mdl = LinearRegression()
        nonlinear_mdl = create_hist_gradient_boost_regressor()
        linear_mdl.fit(x_train, y_train.squeeze())
        nonlinear_mdl.fit(x_train, y_train.squeeze())

        return np.mean((y_test - shape_into_2d(linear_mdl.predict(x_test))) ** 2) <= np.mean(
            (y_test - shape_into_2d(nonlinear_mdl.predict(x_test))) ** 2
        )


def find_best_model(
    prediction_model_factories: List[Callable[[], PredictionModel]],
    X: np.ndarray,
    Y: np.ndarray,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_samples_per_split: int = 10000,
    model_selection_splits: int = 5,
    n_jobs: Optional[int] = None,
) -> Callable[[], PredictionModel]:
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    X, Y = shape_into_2d(X, Y)

    is_classification_problem = isinstance(prediction_model_factories[0](), ClassificationModel)

    if metric is None:
        if is_classification_problem:
            metric = lambda y_true, y_preds: -metrics.f1_score(
                y_true, y_preds, average="macro", zero_division=0
            )  # Higher score is better
        else:
            metric = metrics.mean_squared_error

    labelBinarizer = None
    if is_classification_problem:
        labelBinarizer = MultiLabelBinarizer()
        labelBinarizer.fit(Y)

    kfolds = list(KFold(n_splits=model_selection_splits).split(range(X.shape[0])))

    def estimate_average_score(prediction_model_factory: Callable[[], PredictionModel], random_seed: int) -> float:
        set_random_seed(random_seed)

        average_result = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for train_indices, test_indices in kfolds:
                model_instance = prediction_model_factory()
                model_instance.fit(X[train_indices[:max_samples_per_split]], Y[train_indices[:max_samples_per_split]])

                y_true = Y[test_indices[:max_samples_per_split]]
                y_pred = model_instance.predict(X[test_indices[:max_samples_per_split]])
                if labelBinarizer is not None:
                    y_true = labelBinarizer.transform(y_true)
                    y_pred = labelBinarizer.transform(y_pred)

                average_result += metric(y_true, y_pred)

        return average_result / model_selection_splits

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(prediction_model_factories))
    average_metric_scores = Parallel(n_jobs=n_jobs)(
        delayed(estimate_average_score)(prediction_model_factory, int(random_seed))
        for prediction_model_factory, random_seed in zip(prediction_model_factories, random_seeds)
    )

    return sorted(zip(prediction_model_factories, average_metric_scores), key=lambda x: x[1])[0][0]
