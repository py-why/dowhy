import networkx as nx
import numpy as np
import pandas as pd
from _pytest.python_api import approx
from flaky import flaky
from scipy import stats
from sklearn.metrics import mean_squared_error

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    InvertibleStructuralCausalModel,
    ScipyDistribution,
    fit,
    kernel_based,
)
from dowhy.gcm.auto import assign_causal_mechanisms
from dowhy.gcm.ml import (
    create_hist_gradient_boost_classifier,
    create_hist_gradient_boost_regressor,
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_logistic_regression_classifier,
)
from dowhy.gcm.model_evaluation import (
    EvaluateCausalModelConfig,
    _estimate_conditional_expectations,
    _evaluate_invertibility_assumptions,
    crps,
    evaluate_causal_model,
    nmse,
)


def test_given_good_fit_when_estimate_nrmse_then_returns_zero():
    X = np.random.normal(0, 1, 1000)
    Y = 2 * X

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([2]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=0),
    )

    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=True) == approx(0, abs=0.01)
    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=False) == approx(0, abs=0.01)


def test_given_bad_fit_when_estimate_nrmse_then_returns_high_value():
    X = np.random.normal(0, 1, 1000)
    Y = 2 * X

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([20]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=0),
    )

    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=True) > 1
    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=False) > 1


def test_given_good_fit_but_noisy_data_when_estimate_nrmse_then_returns_expected_result():
    X = np.random.normal(0, 1, 2000)
    Y = 2 * X + np.random.normal(0, 2, 2000)

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([2]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=2),
    )

    # The MSE should be 4 due to the variance of the noise. The RMSE is accordingly 2 / std(Y).
    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=True) == approx(
        4 / np.var(Y), abs=0.05
    )
    assert nmse(Y, _estimate_conditional_expectations(mdl, X, False, 1), squared=False) == approx(
        2 / np.std(Y), abs=0.05
    )


def test_given_good_fit_with_deterministic_data_when_estimate_crps_then_returns_zero():
    X = np.random.normal(0, 1, 1000)
    Y = 2 * X

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([2]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=0),
    )

    assert crps(X, Y, mdl.draw_samples) == approx(0, abs=0.01)


def test_given_bad_fit_with_deterministic_data_when_estimate_crps_then_returns_expected_result():
    X = np.random.normal(0, 1, 2000)
    Y = X

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([1]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=2),
    )

    assert crps(X, Y, mdl.draw_samples) == approx(0.47, abs=0.05)


def test_given_good_fit_but_noisy_data_when_estimate_crps_then_returns_expected_result():
    X = np.random.normal(0, 1, 2000)
    Y = 2 * X + np.random.normal(0, 1, 2000)

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([2]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=1),
    )

    assert crps(X, Y, mdl.draw_samples) == approx(0.26, abs=0.05)


def test_given_very_bad_fit_with_deterministic_data_when_estimate_crps_then_returns_expected_result():
    X = np.random.normal(0, 1, 2000)
    Y = X

    mdl = AdditiveNoiseModel(
        create_linear_regressor_with_given_parameters(np.array([100]), intercept=0),
        noise_model=ScipyDistribution(stats.norm, loc=0, scale=2),
    )

    assert crps(X, Y, mdl.draw_samples) > 1


def test_given_categorical_data_and_a_good_fit_with_deterministic_data_when_estimate_crps_then_returns_zero():
    X = np.random.normal(0, 1, 1000)
    Y = (X > 0).astype(str)

    mdl = ClassifierFCM(create_logistic_regression_classifier())
    mdl.fit(X, Y)

    X = np.random.normal(0, 1, 1000)
    Y = (X > 0).astype(str)

    assert crps(X, Y, mdl.draw_samples) == approx(0.02, abs=0.01)


def test_given_categorical_data_and_a_bad_fit_with_deterministic_data_when_estimate_crps_then_returns_expected_result():
    X = np.random.normal(0, 1, 1000)
    Y = (X > 0).astype(str)

    mdl = ClassifierFCM(create_logistic_regression_classifier())
    mdl.fit(X, Y)

    X = np.random.normal(0, 1, 1000)
    Y = ((X + 1) > 0).astype(str)

    assert crps(X, Y, mdl.draw_samples) == approx(0.3, abs=0.05)


@flaky(max_runs=3)
def test_given_multiplicative_noise_data_when_evaluate_invertibility_assumptions_then_rejects():
    X0 = np.random.normal(0, 1, 5000)
    Y = X0 * np.random.normal(0, 0.1, 5000)
    data = pd.DataFrame({"X0": X0, "Y": Y})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "Y")]))

    assign_causal_mechanisms(causal_model, data)
    causal_model.set_causal_mechanism("Y", AdditiveNoiseModel(create_linear_regressor()))
    fit(causal_model, data)

    assert _evaluate_invertibility_assumptions(causal_model, data, kernel_based, 0.05, None, 1)["Y"][1]


@flaky(max_runs=3)
def test_given_additive_noise_data_when_evaluate_invertibility_assumptions_then_does_not_reject():
    X0 = np.random.normal(0, 1, 5000)
    Y = X0 + np.random.normal(0, 0.1, 5000)
    data = pd.DataFrame({"X0": X0, "Y": Y})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "Y")]))

    assign_causal_mechanisms(causal_model, data)
    causal_model.set_causal_mechanism("Y", AdditiveNoiseModel(create_linear_regressor()))
    fit(causal_model, data)

    assert not _evaluate_invertibility_assumptions(causal_model, data, kernel_based, 0.05, None, 1)["Y"][1]


@flaky(max_runs=3)
def test_given_continuous_data_only_when_evaluate_model_returns_expected_information():
    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.normal(0, 1, 1000)
    Y = X0 + X1 + np.random.normal(0, 0.1, 1000)

    data = pd.DataFrame({"X0": X0, "X1": X1, "Y": Y})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "Y"), ("X1", "Y")]))

    assign_causal_mechanisms(causal_model, data)
    fit(causal_model, data)

    summary = evaluate_causal_model(
        causal_model,
        data,
        compare_mechanism_baselines=True,
        config=EvaluateCausalModelConfig(
            baseline_models_regression=[create_linear_regressor, create_hist_gradient_boost_regressor],
        ),
    )

    assert summary.overall_kl_divergence == approx(0, abs=0.05)

    assert summary.mechanism_performances["X0"].kl_divergence == approx(0, abs=0.2)
    assert summary.mechanism_performances["X0"].crps == None
    assert summary.mechanism_performances["X0"].nmse == None
    assert summary.mechanism_performances["X0"].r2 == None
    assert summary.mechanism_performances["X0"].f1 == None
    assert summary.mechanism_performances["X0"].total_number_baselines == 0

    assert summary.mechanism_performances["X1"].kl_divergence == approx(0, abs=0.2)
    assert summary.mechanism_performances["X1"].crps == None
    assert summary.mechanism_performances["X1"].nmse == None
    assert summary.mechanism_performances["X1"].r2 == None
    assert summary.mechanism_performances["X1"].f1 == None
    assert summary.mechanism_performances["X0"].total_number_baselines == 0

    assert summary.mechanism_performances["Y"].kl_divergence == None
    assert summary.mechanism_performances["Y"].crps == approx(0.05, abs=0.02)
    assert summary.mechanism_performances["Y"].nmse == approx(0.07, abs=0.03)
    assert summary.mechanism_performances["Y"].r2 == approx(1, abs=0.05)
    assert summary.mechanism_performances["Y"].f1 == None
    assert 0 < summary.mechanism_performances["Y"].total_number_baselines <= 2
    assert summary.mechanism_performances["Y"].count_better_performance == 0

    assert "X0" not in summary.pnl_assumptions
    assert "X1" not in summary.pnl_assumptions
    assert not summary.pnl_assumptions["Y"][1]
    assert summary.pnl_assumptions["Y"][2] == 0.05

    summary.plot_falsification_histogram = False
    summary_string = str(summary)

    assert (
        """Evaluated the performance of the causal mechanisms and the invertibility assumption of the causal mechanisms and the overall average KL divergence between generated and observed distribution and the graph structure. The results are as follows:

==== Evaluation of Causal Mechanisms ====
The used evaluation metrics are:
- KL divergence (only for root-nodes): Evaluates the divergence between the generated and the observed distribution.
- Mean Squared Error (MSE): Evaluates the average squared differences between the observed values and the conditional expectation of the causal mechanisms.
- Normalized MSE (NMSE): The MSE normalized by the standard deviation for better comparison.
- R2 coefficient: Indicates how much variance is explained by the conditional expectations of the mechanisms. Note, however, that this can be misleading for nonlinear relationships.
- F1 score (only for categorical non-root nodes): The harmonic mean of the precision and recall indicating the goodness of the underlying classifier model.
- (normalized) Continuous Ranked Probability Score (CRPS): The CRPS generalizes the Mean Absolute Percentage Error to probabilistic predictions. This gives insights into the accuracy and calibration of the causal mechanisms.
NOTE: Every metric focuses on different aspects and they might not consistently indicate a good or bad performance.
We will mostly utilize the CRPS for comparing and interpreting the performance of the mechanisms, since this captures the most important properties for the causal model."""
        in summary_string
    )
    assert "--- Node X0\n" "- The KL divergence between generated and observed distribution is " in summary_string
    assert "--- Node X1\n" "- The KL divergence between generated and observed distribution is " in summary_string
    assert "--- Node Y\n" "- The MSE is " in summary_string
    assert "- The NMSE is " in summary_string
    assert "- The R2 coefficient is " in summary_string
    assert "- The normalized CRPS is " in summary_string
    assert "The estimated CRPS indicates a very good model performance." in summary_string

    assert "The mechanism is better or equally good than all " in summary_string
    assert "==== Evaluation of Invertible Functional Causal Model Assumption ====" in summary_string
    assert (
        "Note that these results are based on statistical independence tests, and the fact that the assumption was "
        "not rejected does not necessarily imply that it is correct. There is just no evidence against it."
        in summary_string
    )

    assert "==== Evaluation of Generated Distribution ====" in summary_string
    assert (
        "The estimated KL divergence indicates an overall very good representation of the data distribution"
        in summary_string
    )
    assert "==== Evaluation of the Causal Graph Structure ====" in summary_string
    assert (
        """==== NOTE ====
Always double check the made model assumptions with respect to the graph structure and choice of causal mechanisms.
All these evaluations give some insight into the goodness of the causal model, but should not be overinterpreted, since some causal relationships can be intrinsically hard to model. Furthermore, many algorithms are fairly robust against misspecifications or poor performances of causal mechanisms."""
        in summary_string
    )


@flaky(max_runs=3)
def test_given_categorical_data_only_when_evaluate_model_returns_expected_information():
    X0 = np.random.normal(0, 1, 2000)
    X1 = np.random.normal(0, 1, 2000)
    Y = (X0 + X1 + np.random.normal(0, 0.1, 2000) > 0).astype(str)

    data = pd.DataFrame({"X0": X0, "X1": X1, "Y": Y})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "Y"), ("X1", "Y")]))

    assign_causal_mechanisms(causal_model, data)
    fit(causal_model, data)

    summary = evaluate_causal_model(
        causal_model,
        data,
        compare_mechanism_baselines=True,
        config=EvaluateCausalModelConfig(
            baseline_models_classification=[
                create_logistic_regression_classifier,
                create_hist_gradient_boost_classifier,
            ],
        ),
    )

    assert summary.overall_kl_divergence == approx(0, abs=0.05)

    assert summary.mechanism_performances["X0"].kl_divergence == approx(0, abs=0.2)
    assert summary.mechanism_performances["X0"].crps == None
    assert summary.mechanism_performances["X0"].nmse == None
    assert summary.mechanism_performances["X0"].r2 == None
    assert summary.mechanism_performances["X0"].f1 == None
    assert summary.mechanism_performances["X0"].total_number_baselines == 0

    assert summary.mechanism_performances["X1"].kl_divergence == approx(0, abs=0.2)
    assert summary.mechanism_performances["X1"].crps == None
    assert summary.mechanism_performances["X1"].nmse == None
    assert summary.mechanism_performances["X1"].r2 == None
    assert summary.mechanism_performances["X1"].f1 == None
    assert summary.mechanism_performances["X0"].total_number_baselines == 0

    assert summary.mechanism_performances["Y"].kl_divergence == None
    assert summary.mechanism_performances["Y"].crps == approx(0.02, abs=0.02)
    assert summary.mechanism_performances["Y"].nmse == None
    assert summary.mechanism_performances["Y"].r2 == None
    assert summary.mechanism_performances["Y"].f1 == approx(0.97, abs=0.05)
    assert 0 < summary.mechanism_performances["Y"].total_number_baselines <= 2
    assert summary.mechanism_performances["Y"].count_better_performance == 0

    assert "X0" not in summary.pnl_assumptions
    assert "X1" not in summary.pnl_assumptions
    assert "Y" not in summary.pnl_assumptions

    summary.plot_falsification_histogram = False
    summary_string = str(summary)

    assert (
        """Evaluated the performance of the causal mechanisms and the invertibility assumption of the causal mechanisms and the overall average KL divergence between generated and observed distribution and the graph structure. The results are as follows:

==== Evaluation of Causal Mechanisms ====
The used evaluation metrics are:
- KL divergence (only for root-nodes): Evaluates the divergence between the generated and the observed distribution.
- Mean Squared Error (MSE): Evaluates the average squared differences between the observed values and the conditional expectation of the causal mechanisms.
- Normalized MSE (NMSE): The MSE normalized by the standard deviation for better comparison.
- R2 coefficient: Indicates how much variance is explained by the conditional expectations of the mechanisms. Note, however, that this can be misleading for nonlinear relationships.
- F1 score (only for categorical non-root nodes): The harmonic mean of the precision and recall indicating the goodness of the underlying classifier model.
- (normalized) Continuous Ranked Probability Score (CRPS): The CRPS generalizes the Mean Absolute Percentage Error to probabilistic predictions. This gives insights into the accuracy and calibration of the causal mechanisms.
NOTE: Every metric focuses on different aspects and they might not consistently indicate a good or bad performance.
We will mostly utilize the CRPS for comparing and interpreting the performance of the mechanisms, since this captures the most important properties for the causal model."""
        in summary_string
    )
    assert "--- Node X0\n" "- The KL divergence between generated and observed distribution is " in summary_string
    assert "--- Node X1\n" "- The KL divergence between generated and observed distribution is " in summary_string
    assert "--- Node Y\n" "- The F1 score is " in summary_string
    assert "- The normalized CRPS is " in summary_string
    assert "The estimated CRPS indicates a very good model performance." in summary_string

    assert "The mechanism is better or equally good than all " in summary_string
    assert "==== Evaluation of Invertible Functional Causal Model Assumption ====" in summary_string
    assert "The causal model has no invertible causal models." in summary_string
    assert "==== Evaluation of Generated Distribution ====" in summary_string
    assert (
        "The estimated KL divergence indicates an overall very good representation of the data distribution"
        in summary_string
    )
    assert "==== Evaluation of the Causal Graph Structure ====" in summary_string
    assert (
        """==== NOTE ====
Always double check the made model assumptions with respect to the graph structure and choice of causal mechanisms.
All these evaluations give some insight into the goodness of the causal model, but should not be overinterpreted, since some causal relationships can be intrinsically hard to model. Furthermore, many algorithms are fairly robust against misspecifications or poor performances of causal mechanisms."""
        in summary_string
    )
