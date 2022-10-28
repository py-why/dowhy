import numpy as np
import pytest
from flaky import flaky
from numpy.matlib import repmat
from pytest import approx

from dowhy.gcm.ml import (
    create_hist_gradient_boost_classifier,
    create_hist_gradient_boost_regressor,
    create_linear_regressor,
    create_logistic_regression_classifier,
)
from dowhy.gcm.stats import estimate_ftest_pvalue, marginal_expectation, quantile_based_fwer
from dowhy.gcm.util.general import geometric_median


@flaky(max_runs=5)
def test_estimate_geometric_median():
    a = np.random.normal(10, 1, 100)
    a = np.hstack([a, np.random.normal(10000, 1, 20)])
    b = np.random.normal(-5, 1, 100)
    b = np.hstack([b, np.random.normal(-10000, 1, 20)])

    gm = geometric_median(np.vstack([a, b]).T)

    assert gm[0] == approx(10, abs=0.5)
    assert gm[1] == approx(-5, abs=0.5)


def test_quantile_based_fwer():
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    assert quantile_based_fwer(p_values, quantile=0.5) == 0.055 / 0.5
    assert quantile_based_fwer(p_values, quantile=0.25) == 0.0325 / 0.25
    assert quantile_based_fwer(p_values, quantile=0.75) == 0.0775 / 0.75

    assert (
        quantile_based_fwer(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1]), quantile=0.5)
        == 0.06 / 0.5
    )
    assert quantile_based_fwer(np.array([0.9, 0.95, 1]), quantile=0.5) == 1
    assert quantile_based_fwer(np.array([0, 0, 0]), quantile=0.5) == 0
    assert quantile_based_fwer(np.array([0.33]), quantile=0.5) == 0.33


def test_given_p_values_with_nans_when_using_quantile_based_fwer_then_ignores_the_nan_values():
    p_values = np.array([0.01, np.nan, 0.02, 0.03, 0.04, 0.05, np.nan, 0.06, 0.07, 0.08, 0.09, 0.1])
    assert quantile_based_fwer(p_values, quantile=0.5) == 0.055 / 0.5


def test_quantile_based_fwer_scaling():
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    p_values_scaling = np.array([2, 2, 1, 2, 1, 3, 1, 2, 4, 1])

    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.5) == approx(0.15)
    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.25) == approx(0.17)
    assert quantile_based_fwer(p_values, p_values_scaling, quantile=0.75) == approx(0.193, abs=0.001)


def test_quantile_based_fwer_raises_error():
    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=0)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), np.array([1, 2]), quantile=0.1)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=1.1)

    with pytest.raises(ValueError):
        assert quantile_based_fwer(np.array([0.1, 0.5, 1]), quantile=-0.5)


def test_marginal_expectation_returns_all_results():
    # Just checking formats, i.e. no need for correlation.
    X = np.random.normal(0, 1, (1000, 3))
    Y = np.random.normal(0, 1, (1000, 1))

    model_all_features = create_linear_regressor()
    model_all_features.fit(X, Y)

    results = marginal_expectation(model_all_features.predict, X, X, [0], return_averaged_results=False)
    assert results.shape[0] == 1000
    assert results.shape[1] == 1000
    assert results.shape[2] == 1


def test_marginal_expectation_returns_reduced_results():
    # Just checking formats, i.e. no need for correlation.
    X = np.random.normal(0, 1, (1000, 3))
    Y = np.random.normal(0, 1, (1000, 1))

    model_all_features = create_linear_regressor()
    model_all_features.fit(X, Y)

    results = marginal_expectation(model_all_features.predict, X, X, [0], return_averaged_results=True)
    assert results.shape[0] == 1000
    assert results.shape[1] == 1


@flaky(max_runs=5)
def test_marginal_expectation_independent_continuous_linear():
    X = np.random.normal(0, 1, (1000, 3))
    Y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2]
    Y = Y.reshape(-1)

    X0 = X[:, 0].reshape(-1, 1)
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    X01 = X[:, :2]

    model_all_features = create_linear_regressor()
    model_all_features.fit(X, Y)
    model_feature_0 = create_linear_regressor()
    model_feature_0.fit(X0, Y)
    model_feature_1 = create_linear_regressor()
    model_feature_1.fit(X1, Y)
    model_feature_2 = create_linear_regressor()
    model_feature_2.fit(X2, Y)
    model_feature_01 = create_linear_regressor()
    model_feature_01.fit(X01, Y)

    assert np.mean(
        np.abs(
            model_feature_0.predict(X0).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [0], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_1.predict(X1).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [1], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_2.predict(X2).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [2], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_01.predict(X01).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [0, 1], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0.2, abs=1)

    assert np.mean(
        np.abs(
            model_feature_0.predict(X0).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [0], feature_perturbation="randomize_columns_jointly"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_1.predict(X1).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [1], feature_perturbation="randomize_columns_jointly"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_2.predict(X2).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [2], feature_perturbation="randomize_columns_jointly"
            )
        )
    ) == approx(0.2, abs=1)
    assert np.mean(
        np.abs(
            model_feature_01.predict(X01).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X, [0, 1], feature_perturbation="randomize_columns_jointly"
            )
        )
    ) == approx(0.2, abs=1)


@flaky(max_runs=5)
def test_marginal_expectation_independent_categorical_linear():
    X = np.random.normal(0, 1, (1000, 3))
    Y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2]
    Y = (Y <= 0).reshape(-1)

    X0 = X[:, 0].reshape(-1, 1)
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    X01 = X[:, :2]

    model_all_features = create_logistic_regression_classifier()
    model_all_features.fit(X, Y)
    model_feature_0 = create_logistic_regression_classifier()
    model_feature_0.fit(X0, Y)
    model_feature_1 = create_logistic_regression_classifier()
    model_feature_1.fit(X1, Y)
    model_feature_2 = create_logistic_regression_classifier()
    model_feature_2.fit(X2, Y)
    model_feature_01 = create_logistic_regression_classifier()
    model_feature_01.fit(X01, Y)

    assert np.sum(
        np.mean(
            np.abs(
                model_feature_0.predict_probabilities(X0)
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X,
                    [0],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_1.predict_probabilities(X1)
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X,
                    [1],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_2.predict_probabilities(X2)
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X,
                    [2],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_01.predict_probabilities(X01)
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X,
                    [0, 1],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)


@flaky(max_runs=5)
def test_marginal_expectation_independent_continuous_nonlinear():
    X = np.random.normal(0, 1, (2000, 3))
    Y = (2 * X[:, 0] + X[:, 1]) ** 2 + X[:, 2]
    Y = Y.reshape(-1)

    X0 = X[:, 0].reshape(-1, 1)
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    X01 = X[:, :2]

    model_all_features = create_hist_gradient_boost_regressor()
    model_all_features.fit(X, Y)
    model_feature_0 = create_hist_gradient_boost_regressor()
    model_feature_0.fit(X0, Y)
    model_feature_1 = create_hist_gradient_boost_regressor()
    model_feature_1.fit(X1, Y)
    model_feature_2 = create_hist_gradient_boost_regressor()
    model_feature_2.fit(X2, Y)
    model_feature_01 = create_hist_gradient_boost_regressor()
    model_feature_01.fit(X01, Y)

    assert np.mean(
        np.abs(
            model_feature_0.predict(X0[:100]).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X[:100], [0], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0, abs=100)
    assert np.mean(
        np.abs(
            model_feature_1.predict(X1[:100]).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X[:100], [1], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0, abs=100)
    assert np.mean(
        np.abs(
            model_feature_2.predict(X2[:100]).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X[:100], [2], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0, abs=100)
    assert np.mean(
        np.abs(
            model_feature_01.predict(X01[:100]).reshape(-1, 1)
            - marginal_expectation(
                model_all_features.predict, X, X[:100], [0, 1], feature_perturbation="randomize_columns_independently"
            )
        )
    ) == approx(0, abs=100)


@flaky(max_runs=5)
def test_marginal_expectation_independent_categorical_nonlinear():
    X = np.random.normal(0, 1, (1000, 3))
    Y = (2 * X[:, 0] + X[:, 1]) ** 2 + X[:, 2]
    Y = (Y <= np.mean(Y)).reshape(-1)

    X0 = X[:, 0].reshape(-1, 1)
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    X01 = X[:, :2]

    model_all_features = create_hist_gradient_boost_classifier()
    model_all_features.fit(X, Y)
    model_feature_0 = create_hist_gradient_boost_classifier()
    model_feature_0.fit(X0, Y)
    model_feature_1 = create_hist_gradient_boost_classifier()
    model_feature_1.fit(X1, Y)
    model_feature_2 = create_hist_gradient_boost_classifier()
    model_feature_2.fit(X2, Y)
    model_feature_01 = create_hist_gradient_boost_classifier()
    model_feature_01.fit(X01, Y)

    assert np.sum(
        np.mean(
            np.abs(
                model_feature_0.predict_probabilities(X0[:100])
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X[:100],
                    [0],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_1.predict_probabilities(X1[:100])
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X[:100],
                    [1],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_2.predict_probabilities(X2[:100])
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X[:100],
                    [2],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)
    assert np.sum(
        np.mean(
            np.abs(
                model_feature_01.predict_probabilities(X01[:100])
                - marginal_expectation(
                    model_all_features.predict_probabilities,
                    X,
                    X[:100],
                    [0, 1],
                    feature_perturbation="randomize_columns_independently",
                )
            ),
            axis=0,
        )
    ) == approx(0.2, abs=1)


def test_given_different_batch_sizes_when_estimating_marginal_expectation_then_returns_expected_result():
    X = np.random.normal(0, 1, (34, 3))
    feature_samples = np.random.normal(0, 1, (123, 3))
    expected_non_aggregated = np.array([repmat(X[i, :], feature_samples.shape[0], 1) for i in range(X.shape[0])])

    def my_pred_func(X: np.ndarray) -> np.ndarray:
        return X.copy()

    assert marginal_expectation(my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=1) == approx(X)
    assert marginal_expectation(my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=10) == approx(X)
    assert marginal_expectation(my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=100) == approx(X)
    assert marginal_expectation(my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=1000) == approx(X)
    assert marginal_expectation(
        my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=feature_samples.shape[0]
    ) == approx(X)

    assert marginal_expectation(
        my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=1, return_averaged_results=False
    ) == approx(expected_non_aggregated)
    assert marginal_expectation(
        my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=10, return_averaged_results=False
    ) == approx(expected_non_aggregated)
    assert marginal_expectation(
        my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=100, return_averaged_results=False
    ) == approx(expected_non_aggregated)
    assert marginal_expectation(
        my_pred_func, feature_samples, X, [0, 1, 2], max_batch_size=1000, return_averaged_results=False
    ) == approx(expected_non_aggregated)
    assert marginal_expectation(
        my_pred_func,
        feature_samples,
        X,
        [0, 1, 2],
        max_batch_size=feature_samples.shape[0],
        return_averaged_results=False,
    ) == approx(expected_non_aggregated)


@flaky(max_runs=2)
def test_given_linear_dependent_data_when_estimate_ftest_pvalue_then_returns_expected_result():
    X_training = np.random.normal(0, 1, 1000)
    Y_training = X_training + np.random.normal(0, 0.05, 1000)

    X_test = np.random.normal(0, 1, 1000)
    Y_test = X_test + np.random.normal(0, 0.05, 1000)

    assert estimate_ftest_pvalue(X_training, np.array([]), Y_training, X_test, np.array([]), Y_test) < 0.05

    Y_training = np.random.normal(0, 0.05, 1000)
    Y_test = np.random.normal(0, 0.05, 1000)

    assert estimate_ftest_pvalue(X_training, np.array([]), Y_training, X_test, np.array([]), Y_test) >= 0.05


@flaky(max_runs=2)
def test_given_multivariate_dependent_data_when_estimate_ftest_pvalue_then_returns_expected_result():
    X1_training = np.random.normal(0, 1, 1000)
    X2_training = np.random.normal(0, 1, 1000)
    Y_training = X1_training + X2_training + np.random.normal(0, 0.05, 1000)

    X1_test = np.random.normal(0, 1, 1000)
    X2_test = np.random.normal(0, 1, 1000)
    Y_test = X1_test + X2_test + np.random.normal(0, 0.05, 1000)

    assert (
        estimate_ftest_pvalue(
            np.column_stack([X1_training, X2_training]),
            X1_training,
            Y_training,
            np.column_stack([X1_test, X2_test]),
            X1_test,
            Y_test,
        )
        < 0.05
    )

    Y_training = X1_training + np.random.normal(0, 0.05, 1000)
    Y_test = X1_test + np.random.normal(0, 0.05, 1000)

    assert (
        estimate_ftest_pvalue(
            np.column_stack([X1_training, X2_training]),
            X1_training,
            Y_training,
            np.column_stack([X1_test, X2_test]),
            X1_test,
            Y_test,
        )
        >= 0.05
    )
