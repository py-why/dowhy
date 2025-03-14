import warnings
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from dowhy.gcm import config
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, ClassifierFCM, DiscreteAdditiveNoiseModel
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
    is_discrete,
    set_random_seed,
    shape_into_2d,
)
from dowhy.graph import get_ordered_predecessors, is_root_node

_LIST_OF_POTENTIAL_CLASSIFIERS_GOOD = [
    partial(create_logistic_regression_classifier, max_iter=10000),
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
    partial(create_lasso_regressor, max_iter=10000),
    create_random_forest_regressor,
    create_support_vector_regressor,
    create_extra_trees_regressor,
    create_knn_regressor,
    create_ada_boost_regressor,
]

_LIST_OF_REGRESSOR_SUPPORTING_MISSING_DATA_GOOD = [create_hist_gradient_boost_regressor]
_LIST_OF_REGRESSOR_SUPPORTING_MISSING_DATA_BETTER = _LIST_OF_REGRESSOR_SUPPORTING_MISSING_DATA_GOOD + [
    create_random_forest_regressor,
    create_extra_trees_regressor,
]

_LIST_OF_CLASSIFIER_SUPPORTING_MISSING_DATA_GOOD = [create_hist_gradient_boost_classifier]
_LIST_OF_CLASSIFIER_SUPPORTING_MISSING_DATA_BETTER = _LIST_OF_CLASSIFIER_SUPPORTING_MISSING_DATA_GOOD + [
    create_random_forest_classifier,
    create_extra_trees_classifier,
]


class AssignmentQuality(Enum):
    GOOD = auto()
    BETTER = auto()
    BEST = auto()


class AutoAssignmentSummary:
    """Summary class for logging and storing information of the auto assignment process."""

    def __init__(self):
        self._nodes: Dict[Dict[Any, Any]] = {}

    def _init_node_entry(self, node: Any):
        if node not in self._nodes:
            self._nodes[node] = {"messages": [], "model_performances": []}

    def add_node_log_message(self, node: Any, message: str):
        self._init_node_entry(node)

        self._nodes[node]["messages"].append(message)

    def add_model_performance(self, node, model: str, performance: str, metric_name: str):
        self._nodes[node]["model_performances"].append((model, performance, metric_name))

    def __str__(self):
        summary_strings = []

        summary_strings.append(
            "When using this auto assignment function, the given data is used to automatically assign a causal "
            "mechanism to each node. Note that causal mechanisms can also be customized and assigned manually.\n"
            "The following types of causal mechanisms are considered for the automatic selection:"
        )
        summary_strings.append("\nIf root node:")
        summary_strings.append(
            "An empirical distribution, i.e., the distribution is represented by randomly sampling from the provided "
            "data. This provides a flexible and non-parametric way to model the marginal distribution and is valid for "
            "all types of data modalities."
        )
        summary_strings.append("\nIf non-root node and the data is continuous:")
        summary_strings.append(
            "Additive Noise Models (ANM) of the form X_i = f(PA_i) + N_i, where PA_i are the "
            "parents of X_i and the unobserved noise N_i is assumed to be independent of PA_i."
            "To select the best model for f, different regression models are evaluated and the model "
            "with the smallest mean squared error is selected."
            "Note that minimizing the mean squared error here is equivalent to selecting the best "
            "choice of an ANM."
        )
        summary_strings.append("\nIf non-root node and the data is discrete:")
        summary_strings.append(
            "Discrete Additive Noise Models have almost the same definition as non-discrete ANMs, but come with an "
            "additional constraint for f to only return discrete values.\n"
            "Note that 'discrete' here refers to numerical values with an order. If the data is categorical, consider "
            "representing them as strings to ensure proper model selection."
        )
        summary_strings.append("\nIf non-root node and the data is categorical:")
        summary_strings.append(
            "A functional causal model based on a classifier, i.e., X_i = f(PA_i, N_i).\n"
            "Here, N_i follows a uniform distribution on [0, 1] and is used to randomly sample a "
            "class (category) using the conditional probability distribution produced by a "
            "classification model."
            "Here, different model classes are evaluated using the (negative) F1 score and the best"
            " performing model class is selected."
        )
        summary_strings.append("\nIn total, %d nodes were analyzed:" % len(list(self._nodes)))

        for node in self._nodes:
            summary_strings.append("\n--- Node: %s" % node)
            summary_strings.extend(self._nodes[node]["messages"])

            if len(self._nodes[node]["model_performances"]) > 0:
                summary_strings.append(
                    "For the model selection, the following models were evaluated on the %s metric:"
                    % self._nodes[node]["model_performances"][0][2]
                )

                for model, performance, metric_name in self._nodes[node]["model_performances"]:
                    summary_strings.append("%s: %s" % (str(model()).replace("()", ""), str(performance)))

        summary_strings.append(
            "\n===Note===\nNote, based on the selected auto assignment quality, the set of " "evaluated models changes."
        )
        summary_strings.append(
            "For more insights toward the quality of the fitted graphical causal model, consider "
            "using the evaluate_causal_model function after fitting the causal mechanisms."
        )
        return "\n".join(summary_strings)


def assign_causal_mechanisms(
    causal_model: ProbabilisticCausalModel,
    based_on: pd.DataFrame,
    quality: AssignmentQuality = AssignmentQuality.GOOD,
    override_models: bool = False,
    experimental_allow_nans: bool = False,
) -> AutoAssignmentSummary:
    """Automatically assigns appropriate causal mechanisms to nodes. If causal mechanisms are already assigned to nodes
    and override_models is set to False, this function only validates the assignments with respect to the graph
    structure. This is, the validation checks whether root nodes have StochasticModels and non-root
    ConditionalStochasticModels assigned.

    The following types of causal mechanisms are considered for the automatic selection:

    If root node:
    An empirical distribution, i.e., the distribution is represented by randomly sampling from the provided data.
    This provides a flexible and non-parametric way to model the marginal distribution and is valid for all types of
    data modalities.

    If non-root node and the data is continuous:
    Additive Noise Models (ANM) of the form X_i = f(PA_i) + N_i, where PA_i are the parents of X_i and the unobserved
    noise N_i is assumed to be independent of PA_i. To select the best model for f, different regression models are
    evaluated and the model with the smallest mean squared error is selected. Note that minimizing the mean squared
    error here is equivalent to selecting the best choice of an ANM. See the following paper for more details:
        Hoyer, P., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2008).
        Nonlinear causal discovery with additive noise models.
        Advances in neural information processing systems, 21

    If non-root node and the data is discrete:
    Discrete Additive Noise Models have almost the same definition as non-discrete ANMs, but come with an additional
    constraint to return discrete values. Note that 'discrete' here refers to numerical values with an order. If the
    data is categorical, consider representing them as strings to ensure proper model selection. See the following
    paper for more details:
        Peters, J., Janzing, D., & Scholkopf, B. (2011).
        Causal inference on discrete data using additive noise models.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(12), 2436-2450.

    If non-root node and the data is categorical:
    A functional causal model based on a classifier, i.e., X_i = f(PA_i, N_i).
    Here, N_i follows a uniform distribution on [0, 1] and is used to randomly sample a class (category) using the
    conditional probability distribution produced by a classification model. Here, different model classes are evaluated
    using the (negative) F1 score and the best performing model class is selected.

    The current model zoo is:

    With "GOOD" quality:
        Numerical:
        - Linear Regressor
        - Linear Regressor with polynomial features
        - Histogram Gradient Boost Regressor

        Categorical:
        - Logistic Regressor
        - Logistic Regressor with polynomial features
        - Histogram Gradient Boost Classifier

    With "BETTER" quality:
        Numerical:
        - Linear Regressor
        - Linear Regressor with polynomial features
        - Gradient Boost Regressor
        - Ridge Regressor
        - Lasso Regressor
        - Random Forest Regressor
        - Support Vector Regressor
        - Extra Trees Regressor
        - KNN Regressor
        - Ada Boost Regressor

        Categorical:
        - Logistic Regressor
        - Logistic Regressor with polynomial features
        - Histogram Gradient Boost Classifier
        - Random Forest Classifier
        - Extra Trees Classifier
        - Support Vector Classifier
        - KNN Classifier
        - Gaussian Naive Bayes Classifier
        - Ada Boost Classifier

    With "BEST" quality:
    An auto ML model based on AutoGluon (optional dependency, needs to be installed).

    :param causal_model: The causal model to whose nodes to assign causal models.
    :param based_on: Jointly sampled data corresponding to the nodes of the given graph.
    :param quality: AssignmentQuality for the automatic model selection and model accuracy. This changes the type of
                    prediction model and time spent on the selection. See the docstring for a list of potential models.
                    The options for the quality are:
        - AssignmentQuality.GOOD: Only a small set of models are evaluated.
            Model selection speed: Fast
            Model training speed: Fast
            Model inference speed: Fast
            Model accuracy: Medium
        - AssignmentQuality.BETTER: A larger set of models are evaluated.
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
    :param override_models: If set to True, existing mechanism assignments are replaced with automatically selected
                            ones. If set to False, the assigned mechanisms are only validated with respect to the graph
                            structure.
    :param experimental_allow_nans: If True, allows missing data for numerical variables. This is an experimental
                                    feature. Not all GCM methods support missing data yet.
    :return: A summary object containing details about the model selection process.
    """
    if not experimental_allow_nans and pd.isna(based_on).any().any():
        raise ValueError(
            "Data contains NaN! This is currently only supported when setting experimental_allow_nans to "
            "True and only for missing data in numerical features. Note that not all GCM features "
            "support missing data yet!"
        )

    auto_assignment_summary = AutoAssignmentSummary()

    for node in nx.topological_sort(causal_model.graph):
        if not override_models and CAUSAL_MECHANISM in causal_model.graph.nodes[node]:
            auto_assignment_summary.add_node_log_message(
                node,
                "Node %s already has a causal mechanism assigned and the override parameter is False. Skipping this "
                "node." % node,
            )
            validate_causal_model_assignment(causal_model.graph, node)
            continue

        model_performances = assign_causal_mechanism_node(causal_model, node, based_on, quality)

        if is_root_node(causal_model.graph, node):
            auto_assignment_summary.add_node_log_message(
                node,
                "Node %s is a root node. Therefore, assigning '%s' to the node representing the marginal distribution."
                % (node, causal_model.causal_mechanism(node)),
            )
        else:
            data_type = "continuous"
            if isinstance(causal_model.causal_mechanism(node), ClassifierFCM):
                data_type = "categorical"
            elif isinstance(causal_model.causal_mechanism(node), DiscreteAdditiveNoiseModel):
                data_type = "discrete"

            auto_assignment_summary.add_node_log_message(
                node,
                "Node %s is a non-root node with %s data. Assigning '%s' to the node."
                % (
                    node,
                    data_type,
                    causal_model.causal_mechanism(node),
                ),
            )

        if isinstance(causal_model.causal_mechanism(node), DiscreteAdditiveNoiseModel):
            auto_assignment_summary.add_node_log_message(
                node,
                "This represents the discrete causal relationship as "
                + str(node)
                + " := f("
                + ",".join([str(parent) for parent in get_ordered_predecessors(causal_model.graph, node)])
                + ") + N.",
            )
        elif isinstance(causal_model.causal_mechanism(node), AdditiveNoiseModel):
            auto_assignment_summary.add_node_log_message(
                node,
                "This represents the causal relationship as "
                + str(node)
                + " := f("
                + ",".join([str(parent) for parent in get_ordered_predecessors(causal_model.graph, node)])
                + ") + N.",
            )
        elif isinstance(causal_model.causal_mechanism(node), ClassifierFCM):
            auto_assignment_summary.add_node_log_message(
                node,
                "This represents the causal relationship as "
                + str(node)
                + " := f("
                + ",".join([str(parent) for parent in get_ordered_predecessors(causal_model.graph, node)])
                + ",N).",
            )

        for model, performance, metric_name in model_performances:
            auto_assignment_summary.add_model_performance(node, model, performance, metric_name)

    return auto_assignment_summary


def assign_causal_mechanism_node(
    causal_model: ProbabilisticCausalModel, node: str, based_on: pd.DataFrame, quality: AssignmentQuality
) -> List[Tuple[Callable[[], PredictionModel], float, str]]:
    if is_root_node(causal_model.graph, node):
        causal_model.set_causal_mechanism(node, EmpiricalDistribution())
        model_performances = []
    else:
        node_data = based_on[node].to_numpy()

        best_model, model_performances = select_model(
            based_on[get_ordered_predecessors(causal_model.graph, node)].to_numpy(),
            node_data,
            quality,
        )

        if isinstance(best_model, ClassificationModel):
            causal_model.set_causal_mechanism(node, ClassifierFCM(best_model))
        else:
            if is_discrete(node_data):
                causal_model.set_causal_mechanism(node, DiscreteAdditiveNoiseModel(best_model))
            else:
                causal_model.set_causal_mechanism(node, AdditiveNoiseModel(best_model))

    return model_performances


def select_model(
    X: np.ndarray, Y: np.ndarray, model_selection_quality: AssignmentQuality
) -> Tuple[Union[PredictionModel, ClassificationModel], List[Tuple[Callable[[], PredictionModel], float, str]]]:
    y_is_categorical = is_categorical(Y)

    if not y_is_categorical:
        y_nan_mask = pd.isna(Y.reshape(-1))

        X = X[~y_nan_mask]
        Y = Y[~y_nan_mask]

    x_has_nans = pd.isna(X).any().any()

    if model_selection_quality == AssignmentQuality.BEST:
        try:
            from dowhy.gcm.ml.autogluon import AutoGluonClassifier, AutoGluonRegressor

            if is_categorical(Y):
                return AutoGluonClassifier(), []
            else:
                return AutoGluonRegressor(), []
        except ImportError:
            raise RuntimeError(
                "AutoGluon module not found! For the BEST auto assign quality, consider installing the "
                "optional AutoGluon dependency."
            )
    elif model_selection_quality == AssignmentQuality.GOOD:
        if x_has_nans:
            list_of_regressor = list(_LIST_OF_REGRESSOR_SUPPORTING_MISSING_DATA_GOOD)
            list_of_classifier = list(_LIST_OF_CLASSIFIER_SUPPORTING_MISSING_DATA_GOOD)
        else:
            list_of_regressor = list(_LIST_OF_POTENTIAL_REGRESSORS_GOOD)
            list_of_classifier = list(_LIST_OF_POTENTIAL_CLASSIFIERS_GOOD)
        model_selection_splits = 5
    elif model_selection_quality == AssignmentQuality.BETTER:
        if x_has_nans:
            list_of_regressor = list(_LIST_OF_REGRESSOR_SUPPORTING_MISSING_DATA_BETTER)
            list_of_classifier = list(_LIST_OF_CLASSIFIER_SUPPORTING_MISSING_DATA_BETTER)
        else:
            list_of_regressor = list(_LIST_OF_POTENTIAL_REGRESSORS_BETTER)
            list_of_classifier = list(_LIST_OF_POTENTIAL_CLASSIFIERS_BETTER)
        model_selection_splits = 5
    else:
        raise ValueError("Invalid model selection quality.")

    if not x_has_nans and auto_apply_encoders(X, auto_fit_encoders(X)).shape[1] <= 5:
        # Avoid too many features
        list_of_regressor += [create_polynom_regressor]
        list_of_classifier += [partial(create_polynom_logistic_regression_classifier, max_iter=10000)]

    if y_is_categorical:
        best_model, model_performances = find_best_model(
            list_of_classifier, X, Y, model_selection_splits=model_selection_splits
        )
        return best_model(), model_performances
    else:
        best_model, model_performances = find_best_model(
            list_of_regressor, X, Y, model_selection_splits=model_selection_splits
        )
        return best_model(), model_performances


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
    max_samples_per_split: int = 20000,
    model_selection_splits: int = 5,
    n_jobs: Optional[int] = None,
) -> Tuple[Callable[[], PredictionModel], List[Tuple[Callable[[], PredictionModel], float, str]]]:
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    X, Y = shape_into_2d(X, Y)

    is_classification_problem = isinstance(prediction_model_factories[0](), ClassificationModel)

    metric_name = "given"

    if metric is None:
        metric_name = "(negative) F1"
        if is_classification_problem:
            metric = lambda y_true, y_preds: -metrics.f1_score(
                y_true, y_preds, average="macro", zero_division=0
            )  # Higher score is better
        else:
            metric_name = "mean squared error (MSE)"
            metric = metrics.mean_squared_error

    labelBinarizer = None
    if is_classification_problem:
        labelBinarizer = MultiLabelBinarizer()
        labelBinarizer.fit(Y)

    if is_classification_problem:
        if len(np.unique(Y)) == 1:
            raise ValueError(
                "The given target samples have only one class! To fit a classification model, there "
                "should be at least two classes."
            )
        kfolds = list(StratifiedKFold(n_splits=model_selection_splits, shuffle=True).split(X, Y))
    else:
        kfolds = list(KFold(n_splits=model_selection_splits, shuffle=True).split(range(X.shape[0])))

    def estimate_average_score(prediction_model_factory: Callable[[], PredictionModel], random_seed: int) -> float:
        set_random_seed(random_seed)

        average_result = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for train_indices, test_indices in kfolds:
                if is_classification_problem and len(np.unique(Y[train_indices[:max_samples_per_split]])) == 1:
                    continue

                model_instance = prediction_model_factory()
                model_instance.fit(X[train_indices[:max_samples_per_split]], Y[train_indices[:max_samples_per_split]])

                y_true = Y[test_indices[:max_samples_per_split]]
                y_pred = model_instance.predict(X[test_indices[:max_samples_per_split]])
                if labelBinarizer is not None:
                    y_true = labelBinarizer.transform(y_true)
                    y_pred = labelBinarizer.transform(y_pred)

                average_result.append(metric(y_true, y_pred))

        if len(average_result) == 0:
            return float("inf")
        else:
            return float(np.mean(average_result))

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(prediction_model_factories))
    average_metric_scores = Parallel(n_jobs=n_jobs)(
        delayed(estimate_average_score)(prediction_model_factory, int(random_seed))
        for prediction_model_factory, random_seed in zip(prediction_model_factories, random_seeds)
    )
    sorted_results = sorted(
        zip(prediction_model_factories, average_metric_scores, [metric_name] * len(prediction_model_factories)),
        key=lambda x: x[1],
    )

    return sorted_results[0][0], sorted_results
