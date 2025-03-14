from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mode
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from dowhy.gcm import (
    ClassifierFCM,
    PostNonlinearModel,
    PredictionModel,
    ProbabilisticCausalModel,
    config,
    draw_samples,
    kernel_based,
)
from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, ConditionalStochasticModel, InvertibleFunctionalCausalModel
from dowhy.gcm.divergence import auto_estimate_kl_divergence
from dowhy.gcm.falsify import EvaluationResult, falsify_graph, plot_evaluation_results
from dowhy.gcm.ml import (
    SklearnRegressionModel,
    create_hist_gradient_boost_classifier,
    create_hist_gradient_boost_regressor,
    create_lasso_regressor,
    create_linear_regressor,
    create_logistic_regression_classifier,
    create_random_forest_classifier,
    create_random_forest_regressor,
    create_ridge_regressor,
)
from dowhy.gcm.ml.classification import (
    create_ada_boost_classifier,
    create_gaussian_nb_classifier,
    create_knn_classifier,
    create_polynom_logistic_regression_classifier,
)
from dowhy.gcm.ml.regression import create_ada_boost_regressor, create_extra_trees_regressor, create_polynom_regressor
from dowhy.gcm.stats import merge_p_values_fdr
from dowhy.gcm.util.general import is_categorical, set_random_seed, shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node


class EvaluateCausalModelConfig:
    """Config for the causal model evaluation."""

    def __init__(
        self,
        mechanism_evaluation_kfolds: int = 5,
        baseline_models_regression: Optional[List[Callable[[], PredictionModel]]] = None,
        baseline_models_classification: Optional[List[Callable[[], PredictionModel]]] = None,
        independence_test_invertible: Callable[[np.ndarray, np.ndarray], float] = partial(
            kernel_based, use_bootstrap=False
        ),
        significance_level_invertible: float = 0.05,
        fdr_control_method_invertible: Optional[str] = "bonferroni",
        bootstrap_runs_invertible: int = 5,
        max_num_permutations_falsify: int = 50,
        independence_test_falsify: Callable[[np.ndarray, np.ndarray], float] = partial(
            kernel_based, use_bootstrap=False, max_num_samples_run=500
        ),
        conditional_independence_test_falsify: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = partial(
            kernel_based, use_bootstrap=False, max_num_samples_run=500
        ),
        falsify_graph_significance_level: float = 0.2,
        n_jobs: Optional[int] = None,
    ):
        """Parameters for the causal model evaluation method. See the parameter description for more details.

        :param mechanism_evaluation_kfolds: Number of folds for evaluating the causal mechanisms.
        :param baseline_models_regression: Baseline models for continuous nodes. The causal mechanisms assigned to the
                                           nodes in the graph are compared against additive noise models with these
                                           baseline regression models.
        :param baseline_models_classification: Baseline models for categorical nodes. The causal mechanisms assigned to the
                                               nodes in the graph are compared against these baseline models.
        :param independence_test_invertible: A method for testing the independence between inputs and estimated noise of invertible
                                             causal mechanisms. This is used to evaluate whether the made model assumptions hold.
        :param significance_level_invertible: The significance level for rejecting the null hypothesis that inputs and residuals
                                              are independent.
        :param fdr_control_method_invertible: The false discovery rate control method when running multiple hypothesis tests. Note that
                                              we can assume that the tests are independent.
        :param bootstrap_runs_invertible: The independence tests are only run on a small subset of samples. This parameter
                                          indicates how many subsets the tests should be performed on. The resulting p-values are
                                          aggregated using a family-wise error control method.
        :param max_num_permutations_falsify: Number of permutations used for falsifying the given graph structure.
        :param independence_test_falsify: A method for testing the independence between two variables used for falsifying
                                          the given graph structure. Note that the variables can be multivariate.
        :param conditional_independence_test_falsify: A method for testing the independence between two variables given a
                                                      third one used for falsifying the given graph structure. Note that
                                                      the variables can be multivariate.
        :param falsify_graph_significance_level: Significance level for rejecting the given graph based on the
                                                 permutation tests. The default of 0.2 here is higher than the usual
                                                 0.05. Consider reducing it to be more strict about falsifying the
                                                 graph.
        :param n_jobs: Number of parallel jobs. Whenever the evaluation method supports parallelization, this parameter
                       is used.
        """
        n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

        if baseline_models_regression is None:
            baseline_models_regression = [
                create_linear_regressor,
                create_polynom_regressor,
                create_hist_gradient_boost_regressor,
                create_ridge_regressor,
                partial(create_lasso_regressor, max_iter=10000),
                create_random_forest_regressor,
                create_extra_trees_regressor,
                create_ada_boost_regressor,
            ]

        if baseline_models_classification is None:
            baseline_models_classification = [
                partial(create_logistic_regression_classifier, max_iter=10000),
                partial(create_polynom_logistic_regression_classifier, max_iter=10000),
                create_hist_gradient_boost_classifier,
                create_random_forest_classifier,
                create_gaussian_nb_classifier,
                create_knn_classifier,
                create_ada_boost_classifier,
            ]

        self.mechanism_evaluation_kfolds = mechanism_evaluation_kfolds
        self.baseline_models_regression = baseline_models_regression
        self.baseline_models_classification = baseline_models_classification
        self.independence_test_invertible = independence_test_invertible
        self.significance_level_invertible = significance_level_invertible
        self.fdr_control_method_invertible = fdr_control_method_invertible
        self.bootstrap_runs_invertible = bootstrap_runs_invertible
        self.max_num_permutations_falsify = max_num_permutations_falsify
        self.independence_test_falsify = independence_test_falsify
        self.conditional_independence_test_falsify = conditional_independence_test_falsify
        self.falsify_graph_significance_level = falsify_graph_significance_level
        self.n_jobs = n_jobs


@dataclass
class MechanismPerformanceResult:
    def __init__(
        self,
        node_name: Any,
        is_root: bool,
        crps: Optional[float],
        kl_divergence: Optional[float],
        mse: Optional[float],
        nmse: Optional[float],
        r2: Optional[float],
        f1: Optional[float],
        count_better_performance: Optional[int],
        best_baseline_model: Optional[str],
        total_number_baselines: int,
        best_baseline_performance: Optional[float],
    ):
        self.node_name = node_name
        self.is_root = is_root
        self.crps = crps
        self.kl_divergence = kl_divergence
        self.mse = mse
        self.nmse = nmse
        self.r2 = r2
        self.f1 = f1
        self.count_better_performance = count_better_performance
        self.best_baseline_model = best_baseline_model
        self.total_number_baselines = total_number_baselines
        self.best_baseline_performance = best_baseline_performance


@dataclass
class CausalModelEvaluationResult:
    mechanism_performances: Optional[Dict[str, MechanismPerformanceResult]] = None
    pnl_assumptions: Optional[Dict[Any, Tuple[float, str, Optional[float]]]] = None
    graph_falsification: Optional[EvaluationResult] = None
    overall_kl_divergence: Optional[float] = None
    plot_falsification_histogram: bool = True

    def __str__(self):
        summary_string = "Evaluated"
        if self.mechanism_performances is not None:
            summary_string += " the performance of the causal mechanisms"
        if self.pnl_assumptions is not None:
            summary_string += " and the invertibility assumption of the causal mechanisms"
        if self.overall_kl_divergence is not None:
            summary_string += " and the overall average KL divergence between generated and observed distribution"
        if self.graph_falsification is not None:
            summary_string += " and the graph structure"

        summary_string += ". The results are as follows:"
        summary_strings = [summary_string]

        if self.mechanism_performances is not None:
            summary_strings.append("\n==== Evaluation of Causal Mechanisms ====")
            summary_strings.append(
                "The used evaluation metrics are:\n"
                "- KL divergence (only for root-nodes): Evaluates the divergence between the generated and the observed distribution.\n"
                "- Mean Squared Error (MSE): Evaluates the average squared differences between the observed values and the conditional expectation of the causal mechanisms.\n"
                "- Normalized MSE (NMSE): The MSE normalized by the standard deviation for better comparison.\n"
                "- R2 coefficient: Indicates how much variance is explained by the conditional expectations of the mechanisms. Note, however, that this can be misleading for nonlinear relationships.\n"
                "- F1 score (only for categorical non-root nodes): The harmonic mean of the precision and recall indicating the goodness of the underlying classifier model.\n"
                "- (normalized) Continuous Ranked Probability Score (CRPS): The CRPS generalizes the Mean Absolute Percentage Error to probabilistic predictions. This gives insights into the accuracy and calibration of the causal mechanisms.\n"
                "NOTE: Every metric focuses on different aspects and they might not consistently indicate a good or bad performance.\n"
                "We will mostly utilize the CRPS for comparing and interpreting the performance of the mechanisms, since this captures the most important properties for the causal model."
            )

            for mechanism_performance in self.mechanism_performances.values():
                summary_strings.append("\n--- Node %s" % mechanism_performance.node_name)
                if mechanism_performance.kl_divergence is not None:
                    summary_strings.append(
                        "- The KL divergence between generated and observed distribution is %s."
                        % mechanism_performance.kl_divergence
                    )
                    summary_strings.append(
                        _get_kl_divergence_interpretation_string(mechanism_performance.kl_divergence)
                    )

                if mechanism_performance.mse is not None:
                    summary_strings.append("- The MSE is %s." % mechanism_performance.mse)

                if mechanism_performance.nmse is not None:
                    summary_strings.append("- The NMSE is %s." % mechanism_performance.nmse)

                if mechanism_performance.r2 is not None:
                    summary_strings.append("- The R2 coefficient is %s." % mechanism_performance.r2)

                if mechanism_performance.f1 is not None:
                    summary_strings.append("- The F1 score is %s." % mechanism_performance.f1)

                if mechanism_performance.crps is not None:
                    summary_strings.append("- The normalized CRPS is %s." % mechanism_performance.crps)
                    summary_strings.append(_get_crps_interpretation_string(mechanism_performance.crps))

                if mechanism_performance.total_number_baselines > 0:
                    summary_strings.append(
                        _get_baseline_model_interpretation_string(
                            mechanism_performance.count_better_performance,
                            mechanism_performance.total_number_baselines,
                            mechanism_performance.best_baseline_model,
                            mechanism_performance.best_baseline_performance,
                        )
                    )

        if self.pnl_assumptions is not None:
            summary_strings.append("\n==== Evaluation of Invertible Functional Causal Model Assumption ====")
            if len(self.pnl_assumptions) == 0:
                summary_strings.append("The causal model has no invertible causal models.")
            else:
                for node in self.pnl_assumptions:
                    rejected = "rejected" if self.pnl_assumptions[node][1] else "not rejected"
                    summary_strings.append(
                        "\n--- The model assumption for node %s is %s with a p-value of %s "
                        "(after potential adjustment) and a significance level of %s."
                        % (node, rejected, self.pnl_assumptions[node][0], self.pnl_assumptions[node][2])
                    )
                    if not self.pnl_assumptions[node][1]:
                        summary_strings.append("This implies that the model assumption might be valid.")
                    else:
                        summary_strings.append(
                            "This implies that the model assumption might not be valid. This is, "
                            "the relationship cannot be represent with this type of mechanism or "
                            "there is a hidden confounder between the node and its parents."
                        )

                summary_strings.append(
                    "\nNote that these results are based on statistical independence tests, and the "
                    "fact that the assumption was not rejected does not necessarily imply that it "
                    "is correct. There is just no evidence against it."
                )

        if self.overall_kl_divergence is not None:
            summary_strings.append("\n==== Evaluation of Generated Distribution ====")
            summary_strings.append(
                "The overall average KL divergence between the generated and observed distribution is %s"
                % self.overall_kl_divergence
            )
            summary_strings.append(_get_kl_divergence_interpretation_string(self.overall_kl_divergence))

        if self.graph_falsification is not None:
            summary_strings.append("\n==== Evaluation of the Causal Graph Structure ====")
            summary_strings.append(str(self.graph_falsification))
            if self.plot_falsification_histogram:
                plot_evaluation_results(self.graph_falsification)

        summary_strings.append("\n==== NOTE ====")
        summary_strings.append(
            "Always double check the made model assumptions with respect to the graph structure and "
            "choice of causal mechanisms."
        )
        summary_strings.append(
            "All these evaluations give some insight into the goodness of the causal model, but should not be "
            "overinterpreted, since some causal relationships can be intrinsically hard to model. Furthermore, many "
            "algorithms are fairly robust against misspecifications or poor performances of causal mechanisms."
        )

        return "\n".join(summary_strings)


def evaluate_causal_model(
    causal_model: ProbabilisticCausalModel,
    data: pd.DataFrame,
    max_num_samples: int = -1,
    evaluate_causal_mechanisms: bool = True,
    compare_mechanism_baselines: bool = False,
    evaluate_invertibility_assumptions: bool = True,
    evaluate_overall_kl_divergence: bool = True,
    evaluate_causal_structure: bool = True,
    config: Optional[EvaluateCausalModelConfig] = None,
) -> CausalModelEvaluationResult:
    """Evaluates the given causal model by running different evaluations.

    Evaluation of Causal Mechanisms:
    The quality of the causal mechanisms is assessed using k-fold cross validation. This means that the models are trained
    from scratch multiple times, which might take a significant amount of time for larger models. Within each fold, the models
    are assessed by different metrics. For all models, the continuous ranked probability score (CRPS) normalized by the
    standard deviation is estimated, an important metric that provides insights to the model performance as well as its
    calibration. Further, if the node is numerical, the mean squared error (MSE), the normalized MSE (normalized by
    the variance) and the R2 coefficient is computed. In case of categorical nodes, the F1 score is computed instead.
    Optionally, the mechanisms' CRPS are compared with baseline models to see if there are baseline models performing significantly better.

    Evaluation of Invertible Functional Causal Model Assumption:
    Invertible causal mechanisms rely on the assumption that the inputs are independent of the reconstructed noise.
    This is, assuming there are no hidden confounders, the noise should be independent of parents of a node. This
    can be evaluated by testing statistical independence between the reconstructed noise and the used input samples.

    Evaluation of Generated Distribution:
    The distribution generated by the causal model is compared with the observed data using KL divergence. To avoid
    estimating the KL divergence of high-dimensional data, we approximate it by calculating the mean KL divergence
    across the individual marginal KL divergences for each node.

    Evaluation of the Causal Graph Structure:
    The causal graph structure is evaluated by running a method to falsify the graph. The method involves conducting
    independence tests and may consume a significant amount of time for more extensive graphs. The results provide
    an indication of whether the graph is rejected or not. It's important to note that a non-rejected graph does not
    guarantee its correctness. It simply means that the evaluation did not find substantial evidence to refute the causal
    graph based on the provided data. However, a rejected graph might indicate potential issues with its structure.

    The outcomes of these evaluation methods should be interpreted with caution, and bad fits should not be
    over-interpreted. Nonetheless, the results can offer insights into the performance of the causal model and potential
    areas for improvement.

    :param causal_model: The causal model to evaluate.
    :param data: The data used for the evaluation.
    :param max_num_samples: The maximum number of samples used for the evaluation. If the runtime is too slow, consider
                            setting this to a smaller value. The default -1 indicates that all samples are used.
    :param evaluate_causal_mechanisms: If True, the causal mechanisms are evaluated.
    :param compare_mechanism_baselines: If True, the causal mechanisms are compared with baseline models to see
                                        if there are model choices that perform significantly better. If False, this
                                        comparison is skipped. This is ignored if evaluate_causal_mechanisms is False.
    :param evaluate_invertibility_assumptions: If True, the model assumption represented by invertible causal mechanisms
                                               is tested.
    :param evaluate_overall_kl_divergence: If True, the KL divergence between the generated and the observed data is
                                           estimated.
    :param evaluate_causal_structure: If True, the causal graph structure is evaluated.
    :return: A summary of the evaluation.
    """
    if config is None:
        config = EvaluateCausalModelConfig()

    evaluation_result = CausalModelEvaluationResult()

    if max_num_samples >= 0 and max_num_samples < data.shape[0]:
        data = data[np.random.choice(data.shape[0], data.shape[0], replace=False)]

    if evaluate_causal_mechanisms:
        evaluation_result.mechanism_performances = _evaluate_model_performances(
            causal_model,
            data,
            compare_mechanism_baselines,
            config.baseline_models_regression,
            config.baseline_models_classification,
            config.mechanism_evaluation_kfolds,
            config.n_jobs,
        )

    if evaluate_invertibility_assumptions:
        evaluation_result.pnl_assumptions = _evaluate_invertibility_assumptions(
            causal_model,
            data,
            config.independence_test_invertible,
            config.significance_level_invertible,
            config.fdr_control_method_invertible,
            config.bootstrap_runs_invertible,
        )

    if evaluate_overall_kl_divergence:
        # Normally, we need to estimate the KL divergence jointly. However, to avoid issues with high dimensional data,
        # we approximate it by taking the average over the marginal KL divergences.
        drawn_samples = draw_samples(causal_model, data.shape[0])

        evaluation_result.overall_kl_divergence = np.mean(
            [
                auto_estimate_kl_divergence(drawn_samples[node].to_numpy(), data[node].to_numpy())
                for node in causal_model.graph.nodes
            ]
        )

    if evaluate_causal_structure:
        evaluation_result.graph_falsification = falsify_graph(
            causal_model.graph,
            data,
            n_permutations=config.max_num_permutations_falsify,
            independence_test=config.independence_test_falsify,
            conditional_independence_test=config.conditional_independence_test_falsify,
            significance_level=config.falsify_graph_significance_level,
            n_jobs=config.n_jobs,
            allow_data_subset=False,
        )

    return evaluation_result


def _evaluate_model_performances(
    causal_model: ProbabilisticCausalModel,
    data: pd.DataFrame,
    compare_mechanism_baselines,
    baseline_models_regression: List[Callable[[], PredictionModel]],
    baseline_models_classification: List[Callable[[], PredictionModel]],
    kfolds: int,
    n_jobs: int,
) -> Dict[Any, Tuple[bool, float, Optional[int], Optional[str], int, Optional[float]]]:
    model_performances = {}

    def evaluate_node(node_name, random_seed):
        set_random_seed(random_seed)

        node_data = data[node_name].to_numpy()
        metric_evaluations = {"CRPS": [], "KL": [], "MSE": [], "NMSE": [], "R2": [], "F1": []}
        baseline_crps = {}
        categorical = is_categorical(node_data)

        if is_root_node(causal_model.graph, node_name):
            for training_indices, test_indices in KFold(n_splits=kfolds, shuffle=True).split(node_data):
                tmp_causal_mechanism = causal_model.causal_mechanism(node_name).clone()
                tmp_causal_mechanism.fit(node_data[training_indices])

                metric_evaluations["KL"].append(
                    auto_estimate_kl_divergence(
                        tmp_causal_mechanism.draw_samples(len(test_indices)), node_data[test_indices]
                    )
                )
        else:
            parent_data = data[get_ordered_predecessors(causal_model.graph, node_name)].to_numpy()

            for training_indices, test_indices in KFold(n_splits=kfolds, shuffle=True).split(parent_data):
                tmp_causal_mechanism = causal_model.causal_mechanism(node_name).clone()
                tmp_causal_mechanism.fit(parent_data[training_indices], node_data[training_indices])

                metric_evaluations["CRPS"].append(
                    crps(parent_data[test_indices], node_data[test_indices], tmp_causal_mechanism.draw_samples)
                )

                conditional_expectations = _estimate_conditional_expectations(
                    tmp_causal_mechanism, parent_data[test_indices], categorical, 50
                )
                if categorical:
                    metric_evaluations["F1"].append(
                        f1_score(node_data[test_indices], conditional_expectations, average="macro", zero_division=0)
                    )
                else:
                    metric_evaluations["MSE"].append(
                        mean_squared_error(node_data[test_indices], conditional_expectations)
                    )
                    metric_evaluations["NMSE"].append(nmse(node_data[test_indices], conditional_expectations))
                    metric_evaluations["R2"].append(r2_score(node_data[test_indices], conditional_expectations))

                if not compare_mechanism_baselines:
                    continue

                if categorical:
                    for baseline_mdl_factory in baseline_models_classification:
                        tmp_classifier_mdl = baseline_mdl_factory()
                        if (
                            isinstance(tmp_causal_mechanism, ClassifierFCM)
                            and isinstance(tmp_causal_mechanism.classifier_model, SklearnRegressionModel)
                            and tmp_causal_mechanism.classifier_model.sklearn_model.__class__
                            == tmp_classifier_mdl.sklearn_model.__class__
                        ):
                            # Do not compare with same model class
                            continue

                        baseline_mechanism = ClassifierFCM(tmp_classifier_mdl)
                        baseline_mechanism.fit(parent_data[training_indices], node_data[training_indices])

                        baseline_crps.setdefault(str(baseline_mechanism), []).append(
                            crps(parent_data[test_indices], node_data[test_indices], baseline_mechanism.draw_samples)
                        )
                else:
                    for baseline_mdl_factory in baseline_models_regression:
                        tmp_reg_mdl = baseline_mdl_factory()
                        if (
                            isinstance(tmp_causal_mechanism, PostNonlinearModel)
                            and isinstance(tmp_causal_mechanism.prediction_model, SklearnRegressionModel)
                            and tmp_causal_mechanism.prediction_model.sklearn_model.__class__
                            == tmp_reg_mdl.sklearn_model.__class__
                        ):
                            # Do not compare with same model class
                            continue

                        baseline_mechanism = AdditiveNoiseModel(tmp_reg_mdl)
                        baseline_mechanism.fit(parent_data[training_indices], node_data[training_indices])

                        baseline_crps.setdefault(str(baseline_mechanism), []).append(
                            crps(parent_data[test_indices], node_data[test_indices], baseline_mechanism.draw_samples)
                        )

        for metric in metric_evaluations:
            metric_evaluations[metric] = (
                float(np.mean(metric_evaluations[metric])) if len(metric_evaluations[metric]) > 0 else None
            )

        count_better_performance = None
        best_baseline_performance = None
        best_baseline_model = None
        total_number_baselines = 0

        if compare_mechanism_baselines:
            count_better_performance = 0

            for k in baseline_crps:
                total_number_baselines += 1
                baseline_crps[k] = float(np.mean(baseline_crps[k]))

                if metric_evaluations["CRPS"] - baseline_crps[k] > 0.05:
                    count_better_performance += 1

                if best_baseline_performance is None:
                    best_baseline_model = k
                    best_baseline_performance = baseline_crps[k]

                if baseline_crps[k] < best_baseline_performance:
                    best_baseline_model = k
                    best_baseline_performance = baseline_crps[k]

        return MechanismPerformanceResult(
            node_name=node_name,
            is_root=is_root_node(causal_model.graph, node_name),
            kl_divergence=metric_evaluations["KL"],
            crps=metric_evaluations["CRPS"],
            mse=metric_evaluations["MSE"],
            nmse=metric_evaluations["NMSE"],
            r2=metric_evaluations["R2"],
            f1=metric_evaluations["F1"],
            count_better_performance=count_better_performance,
            best_baseline_model=best_baseline_model,
            total_number_baselines=total_number_baselines,
            best_baseline_performance=best_baseline_performance,
        )

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(causal_model.graph.nodes))
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_node)(node, int(random_seeds[i]))
        for i, node in enumerate(
            tqdm(
                list(nx.topological_sort(causal_model.graph)),
                position=0,
                leave=True,
                disable=not config.show_progress_bars,
                desc="Evaluating causal mechanisms...",
            )
        )
    )

    return {performance_result.node_name: performance_result for performance_result in all_results}


def _evaluate_invertibility_assumptions(
    causal_model: ProbabilisticCausalModel,
    data: pd.DataFrame,
    independence_test_invertible: Callable[[np.ndarray, np.ndarray], float],
    significance_level_invertible: float,
    fdr_control_method_invertible: Optional[str],
    bootstrap_runs_invertible: int,
    max_num_samples_per_run: int = 2000,
) -> Dict[Any, Tuple[float, bool, float]]:
    all_pnl_p_values = {}

    if max_num_samples_per_run >= data.shape[0]:
        bootstrap_runs_invertible = 1

    max_num_samples_per_run = min(max_num_samples_per_run, data.shape[0])

    for node in causal_model.graph.nodes:
        causal_mechanism = causal_model.causal_mechanism(node)
        if isinstance(causal_mechanism, InvertibleFunctionalCausalModel):
            parent_samples = data[get_ordered_predecessors(causal_model.graph, node)].to_numpy()
            target_samples = data[node].to_numpy()

            tmp_p_values = []
            for run in range(bootstrap_runs_invertible):
                random_indices = np.random.choice(target_samples.shape[0], max_num_samples_per_run, replace=False)
                tmp_p_values.append(
                    independence_test_invertible(
                        causal_mechanism.estimate_noise(target_samples[random_indices], parent_samples[random_indices]),
                        parent_samples[random_indices],
                    )
                )
            all_pnl_p_values[node] = merge_p_values_fdr(tmp_p_values)

    if len(all_pnl_p_values) == 0:
        return all_pnl_p_values

    if fdr_control_method_invertible is not None:
        to_adjust = []
        node_names = []
        for node in all_pnl_p_values:
            node_names.append(node)
            to_adjust.append(all_pnl_p_values[node])

        multiple_test_result = multipletests(
            to_adjust, significance_level_invertible, method=fdr_control_method_invertible
        )

        for i, node in enumerate(node_names):
            all_pnl_p_values[node] = (
                multiple_test_result[1][i],
                multiple_test_result[0][i],
                significance_level_invertible,
            )
    else:
        for node in all_pnl_p_values:
            all_pnl_p_values[node] = (
                all_pnl_p_values[node],
                all_pnl_p_values[node] < significance_level_invertible,
                significance_level_invertible,
            )

    return all_pnl_p_values


def _estimate_conditional_expectations(
    causal_mechanism: ConditionalStochasticModel,
    parent_samples: np.ndarray,
    categorical: bool,
    num_samples_conditional_samples: int,
) -> np.ndarray:
    if isinstance(causal_mechanism, PostNonlinearModel):
        # In case of post non-linear models, we can obtain the conditional expectation directly based on the prediction
        # model. To do this, we can just in pass 0 as the noise, since this would evaluate Y = f(X) + 0 in case of an
        # additive noise model and Y = g(f(X) + 0) in case of a more general model.
        return causal_mechanism.evaluate(parent_samples, np.zeros(parent_samples.shape[0])).reshape(-1)
    elif isinstance(causal_mechanism, ClassifierFCM):
        return causal_mechanism.classifier_model.predict(parent_samples).reshape(-1)
    else:
        if not categorical:
            # Estimate the conditional expectation E[Y | x] by generating multiple samples for Y|x and average them.
            y_preds = np.zeros(parent_samples.shape[0])
            for _ in range(num_samples_conditional_samples):
                y_preds += causal_mechanism.draw_samples(parent_samples).reshape(-1)

            return y_preds / num_samples_conditional_samples
        else:
            # Since these are categorical values, we just need to look for the most frequent element after we drew
            # multiple samples for each input.
            all_draws = []
            for _ in range(num_samples_conditional_samples):
                all_draws.append(causal_mechanism.draw_samples(parent_samples).reshape(-1))

            modes, _ = mode(np.array(all_draws), axis=0)

            return np.array(modes[0].tolist())


def nmse(y_true: np.ndarray, y_pred: np.ndarray, squared: bool = False) -> float:
    """Estimates the Normalized Mean Squared Error (NMSE) based on the given samples. This is, the root mean
    squared error normalized by the variance of the observed values.

    :param y_true: Observed values.
    :param y_pred: Predicted values.
    :param squared: If True, returns the normalized MSE if False, it returns the normalized RMSE.
    :return: The normalized MSE.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    y_std = np.std(y_true)
    mse = mean_squared_error(y_true, y_pred)

    if not squared:
        mse = np.sqrt(mse)

    if y_std == 0:
        return mse

    return mse / (np.var(y_true) if squared else y_std)


def crps(
    X: np.ndarray,
    Y: np.ndarray,
    conditional_sampling_method: Callable[[np.ndarray], np.ndarray],
    num_conditional_samples: int = 100,
    normalize: bool = True,
) -> float:
    """Estimates the (normalized) Continuous Ranked Probability Score (CRPS) based on the given data and generation
    process. This is used to check the calibration of a probabilistic prediction.

    :param X: Observations of the input features.
    :param Y: Observations of the corresponding target value.
    :param conditional_sampling_method: Method to sample from the conditional given an input sample from X.
    :param num_conditional_samples: Number of samples that should be drawn from the conditional to estimate the CRPS.
    :param normalize: If True, the target values are normalized in the continuous case by the standard deviation of the
                      expected Y values. By this, the CRPS become comparable across different scales.
    :return: The Continuous Ranked Probability Score.
    """

    def empirical_crps(generated_Y, observed_y):
        """Estimates \int (F(x) - 1_{x >= y})**2 dx = E[|X - y|] - 1/2 E[|X - X'|]

        Here, X is generated_Y and y is observed_y. The X' are another set of generated_Y, however, we can also take
        the difference over all combinations to estimate the expectation."""
        generated_Y = generated_Y.reshape(-1)
        observed_y = observed_y.squeeze()

        return np.mean(np.abs(generated_Y - observed_y)) - 0.5 * np.mean(
            np.abs(np.subtract.outer(generated_Y, generated_Y))
        )

    if Y.ndim > 1 and Y.shape[1] > 1:
        raise ValueError("Y has to be one dimensional!")

    X = shape_into_2d(X)
    Y = Y.reshape(-1)

    crps_values = []

    categorical = is_categorical(Y)
    if categorical:
        # In the categorical case, this is equivalent to the Brier score. However, the following formulation allows
        # categorical data with more than two classes.
        all_classes = np.unique(Y)

        for x, y in zip(X, Y):
            samples = conditional_sampling_method(np.tile(x, (num_conditional_samples, 1)))

            sample_categorical_crps = []
            for cat in all_classes:
                sample_categorical_crps.append(
                    empirical_crps((samples == cat).astype(int), np.array(y == cat).astype(int))
                )
            crps_values.append(np.mean(sample_categorical_crps))
    else:
        std_Y = 1

        if normalize:
            std_Y = np.std(Y)

            if std_Y == 0:
                std_Y = 1

        for x, y in zip(X, Y):
            crps_values.append(
                empirical_crps(conditional_sampling_method(np.tile(x, (num_conditional_samples, 1))) / std_Y, y / std_Y)
            )

    return float(np.mean(np.array(crps_values)))


def _get_kl_divergence_interpretation_string(kl_divergence: float) -> str:
    if kl_divergence < 0.5:
        return "The estimated KL divergence indicates an overall very good representation of the data distribution."
    elif 0.5 <= kl_divergence < 1:
        return (
            "The estimated KL divergence indicates a good representation of the data distribution, but might "
            "indicate some smaller mismatches between the distributions."
        )
    elif 1 <= kl_divergence < 1.5:
        return "The estimated KL divergence indicates some mismatches between the distributions."
    elif 1.5 <= kl_divergence < 3:
        return "The estimated KL divergence indicates some significant mismatches between the distributions."
    else:
        return (
            "The estimated KL divergence indicates a significant mismatches between the distributions. "
            "Consider using models that better fit the distribution. However, also note that regardless of the model "
            "choice, intrinsically weak connections or very small signal to noise ratios will always lead to a poor "
            "fit."
        )


def _get_crps_interpretation_string(crps: float) -> str:
    if crps < 0.2:
        return "The estimated CRPS indicates a very good model performance."
    elif 0.2 <= crps < 0.35:
        return "The estimated CRPS indicates a good model performance."
    elif 0.35 <= crps < 0.65:
        return (
            "The estimated CRPS indicates only a fair model performance. "
            "Note, however, that a high CRPS could also result from a small signal to noise ratio."
        )
    elif 0.65 <= crps < 0.9:
        return (
            "The estimated CRPS indicates a rather poor model performance. Note, however, that a high CRPS could also "
            "result from a very small signal to noise ratio."
        )
    elif 0.9 <= crps < 1:
        return (
            "The estimated CRPS indicates a bad model performance. Consider trying "
            "alternative causal mechanism types. Note, however, that a high CRPS could also "
            "result from a very small signal to noise ratio."
        )
    else:
        return (
            "The estimated CRPS indicates a very bad model performance. Consider trying alternative causal mechanism "
            "types."
        )


def _get_baseline_model_interpretation_string(
    count_better: int, count_total: int, best_baseline_model: str, best_baseline_performance: float
) -> str:
    percentage_better = count_better / count_total
    best_baseline_model = best_baseline_model.replace("()", "")

    if percentage_better == 0:
        return "The mechanism is better or equally good than all %s baseline mechanisms." % str(count_total)
    elif percentage_better < 0.2:
        return (
            str(count_better)
            + " of "
            + str(count_total)
            + " baseline mechanisms performed significantly better. The best baseline mechanism is: "
            + best_baseline_model
            + " with a CRPS of "
            + str(best_baseline_performance)
            + "."
        )
    elif 0.2 < percentage_better < 0.4:
        return (
            str(count_better)
            + " of "
            + str(count_total)
            + " baseline mechanisms performed significantly better. The best baseline "
            "mechanism is: "
            + best_baseline_model
            + " with a CRPS of "
            + str(best_baseline_performance)
            + ". Accordingly, the current mechanism could be improved using a "
            + best_baseline_model
            + " instead. If you are using an auto assigment, consider a "
            "better assignment quality level such as AssignmentQuality.BETTER."
        )
    else:
        return (
            str(count_better)
            + " of "
            + str(count_total)
            + " baseline mechanisms performed significantly better. The best baseline "
            "mechanism is: "
            + best_baseline_model
            + " with a CRPS of "
            + str(best_baseline_performance)
            + ". Seeing this, consider changing the mechanism to a "
            + best_baseline_model
            + " or, if you are using an auto assigment, consider a better assignment quality "
            "level such as AssignmentQuality.BETTER."
        )
