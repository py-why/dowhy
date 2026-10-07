import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm.shapley import ShapleyConfig
from dowhy.gcm.unit_change import (
    SklearnLinearRegressionModel,
    unit_change,
    unit_change_linear,
    unit_change_linear_input_only,
    unit_change_nonlinear,
    unit_change_nonlinear_input_only,
)


@flaky(max_runs=5)
def test_given_fitted_linear_mechanisms_with_output_change_when_evaluate_unit_change_linear_method_then_returns_correct_attributions():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 2 * A1 + 3 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2))

    background_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A1, B1)), C1)
    )
    foreground_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A2, B2)), C2)
    )

    actual_contributions = unit_change_linear(
        background_mechanism, background_df, foreground_mechanism, foreground_df, ["A", "B"]
    )
    expected_contributions = pd.DataFrame(
        data=dict(
            A=(3 + 2) * (A2 - A1) / 2, B=(2 + 3) * (B2 - B1) / 2, f=(A1 + A2) * (3 - 2) / 2 + (B1 + B2) * (2 - 3) / 2
        )
    )

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


@flaky(max_runs=5)
def test_given_linear_mechanisms_with_different_intercepts_when_evaluate_unit_change_linear_then_attributions_sum_to_output_change():
    # When the mechanisms are fit with intercepts (the sklearn default), the mechanism ('f') attribution must include
    # the intercept difference. Otherwise the per-row attributions do not sum to the true output change.
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 2 * A1 + 3 * B1 + 5.0

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2 - 2.0

    background_df = pd.DataFrame(data=dict(A=A1, B=B1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2))

    background_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=True).fit(np.column_stack((A1, B1)), C1)
    )
    foreground_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=True).fit(np.column_stack((A2, B2)), C2)
    )

    contributions = unit_change_linear(
        background_mechanism, background_df, foreground_mechanism, foreground_df, ["A", "B"]
    )

    true_output_change = (
        foreground_mechanism.predict(foreground_df[["A", "B"]].to_numpy()).flatten()
        - background_mechanism.predict(background_df[["A", "B"]].to_numpy()).flatten()
    )

    np.testing.assert_array_almost_equal(contributions.sum(axis=1).to_numpy(), true_output_change, decimal=5)

    # The linear method must also agree with the (correct) nonlinear Shapley counterpart.
    nonlinear_contributions = unit_change_nonlinear(
        background_mechanism,
        background_df,
        foreground_mechanism,
        foreground_df,
        ["A", "B"],
        shapley_config=ShapleyConfig(n_jobs=1),
    )
    np.testing.assert_array_almost_equal(contributions, nonlinear_contributions, decimal=1)


@flaky(max_runs=5)
def test_given_fitted_linear_mechanisms_with_output_change_when_evaluate_unit_change_linear_and_nonlinear_methods_then_attributions_are_consistent():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 2 * A1 + 3 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2))

    background_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A1, B1)), C1)
    )
    foreground_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A2, B2)), C2)
    )

    actual_contributions = unit_change_linear(
        background_mechanism, background_df, foreground_mechanism, foreground_df, ["A", "B"]
    )
    expected_contributions = unit_change_nonlinear(
        background_mechanism,
        background_df,
        foreground_mechanism,
        foreground_df,
        ["A", "B"],
        shapley_config=ShapleyConfig(n_jobs=1),
    )

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


def test_given_unfitted_mechanisms_when_evaluate_unit_change_methods_then_raises_exception():
    with pytest.raises(NotFittedError):
        unit_change_linear(
            SklearnLinearRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnLinearRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )

    with pytest.raises(NotFittedError):
        unit_change_nonlinear(
            SklearnRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
            shapley_config=ShapleyConfig(n_jobs=1),
        )

    with pytest.raises(NotFittedError):
        unit_change_nonlinear(
            SklearnRegressionModel(RFR(n_jobs=1)),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnRegressionModel(RFR(n_jobs=1)),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
            shapley_config=ShapleyConfig(n_jobs=1),
        )


def test_given_fitted_nonlinnear_mechanisms_when_evaluate_unit_change_linear_method_then_raises_exception():
    with pytest.raises(AttributeError):
        unit_change_linear(
            SklearnRegressionModel(RFR(n_jobs=1).fit(np.random.normal(size=(100, 2)), np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnRegressionModel(RFR(n_jobs=1).fit(np.random.normal(size=(100, 2)), np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )


@flaky(max_runs=5)
def test_given_fitted_mechanisms_with_no_input_change_when_evaluate_unit_change_input_only_methods_then_returns_zero_attributions():
    num_rows = 100
    A = np.random.normal(size=num_rows)
    B = np.random.normal(size=num_rows)
    C = 3 * A + 2 * B

    background_df = pd.DataFrame(data=dict(A=A, B=B, C=C))
    foreground_df = pd.DataFrame(data=dict(A=A, B=B, C=C))

    actual_contributions = unit_change_nonlinear_input_only(
        SklearnRegressionModel(RFR(n_jobs=1).fit(np.column_stack((A, B)), C)),
        background_df,
        foreground_df,
        ["A", "B"],
        shapley_config=ShapleyConfig(n_jobs=1),
    )
    expected_contributions = pd.DataFrame(data=dict(A=np.zeros(num_rows), B=np.zeros(num_rows)))
    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


@flaky(max_runs=5)
def test_given_fitted_linear_mechanism_with_input_change_when_evaluate_unit_change_linear_input_only_then_returns_correct_attributions():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 3 * A1 + 2 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1, C=C1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2, C=C2))

    fitted_linear_reg = LinearRegression()
    fitted_linear_reg.coef_ = np.array([3, 2])

    actual_contributions = unit_change_linear_input_only(
        SklearnLinearRegressionModel(fitted_linear_reg), background_df, foreground_df, ["A", "B"]
    )
    expected_contributions = pd.DataFrame(data=dict(A=3 * (A2 - A1), B=2 * (B2 - B1)))
    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


@flaky(max_runs=5)
def test_given_fitted_linear_mechanism_with_input_change_when_evaluate_unit_change_input_only_methods_then_attributions_are_consistent():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 3 * A1 + 2 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1, C=C1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2, C=C2))

    mechanism = SklearnLinearRegressionModel(LinearRegression().fit(np.column_stack((A1, B1)), C1))
    actual_contributions = unit_change_nonlinear_input_only(
        mechanism, background_df, foreground_df, ["A", "B"], shapley_config=ShapleyConfig(n_jobs=1)
    )
    expected_contributions = unit_change_linear_input_only(mechanism, background_df, foreground_df, ["A", "B"])

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


def test_given_unfitted_mechanisms_when_evaluate_unit_change_input_only_methods_then_raises_exception():
    with pytest.raises(NotFittedError):
        unit_change_linear_input_only(
            SklearnLinearRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )

    with pytest.raises(NotFittedError):
        unit_change_nonlinear_input_only(
            SklearnLinearRegressionModel(LinearRegression()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
            shapley_config=ShapleyConfig(n_jobs=1),
        )

    with pytest.raises(NotFittedError):
        unit_change_nonlinear_input_only(
            SklearnRegressionModel(RFR(n_jobs=1)),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
            shapley_config=ShapleyConfig(n_jobs=1),
        )


def test_given_fitted_nonlinnear_mechanism_when_evaluate_unit_change_linear_input_only_method_then_raises_exception():
    with pytest.raises(AttributeError):
        unit_change_linear_input_only(
            SklearnRegressionModel(RFR(n_jobs=1).fit(np.random.normal(size=(100, 2)), np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )


@flaky(max_runs=5)
def test_given_single_mechanism_with_default_optional_parameters_when_evaluate_unit_change_then_returns_correct_attributions_to_input_only():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 2 * A1 + 3 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    # C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2))

    mechanism = SklearnLinearRegressionModel(LinearRegression(fit_intercept=False).fit(np.column_stack((A1, B1)), C1))

    actual_contributions = unit_change(background_df, foreground_df, ["A", "B"], mechanism)
    expected_contributions = unit_change_linear_input_only(mechanism, background_df, foreground_df, ["A", "B"])

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)

    mechanism = SklearnRegressionModel(RFR(n_jobs=1).fit(np.column_stack((A1, B1)), C1))

    actual_contributions = unit_change(background_df, foreground_df, ["A", "B"], mechanism)
    expected_contributions = unit_change_nonlinear_input_only(
        mechanism, background_df, foreground_df, ["A", "B"], shapley_config=ShapleyConfig(n_jobs=1)
    )
    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


@flaky(max_runs=5)
def test_given_two_mechanisms_when_evaluate_unit_change_then_returns_correct_attributions_to_both_mechanism_and_input():
    num_rows = 100
    A1 = np.random.normal(size=num_rows)
    B1 = np.random.normal(size=num_rows)
    C1 = 2 * A1 + 3 * B1

    A2 = np.random.normal(size=num_rows)
    B2 = np.random.normal(size=num_rows)
    C2 = 3 * A2 + 2 * B2

    background_df = pd.DataFrame(data=dict(A=A1, B=B1))
    foreground_df = pd.DataFrame(data=dict(A=A2, B=B2))

    background_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A1, B1)), C1)
    )
    foreground_mechanism = SklearnLinearRegressionModel(
        LinearRegression(fit_intercept=False).fit(np.column_stack((A2, B2)), C2)
    )

    actual_contributions = unit_change(
        background_df, foreground_df, ["A", "B"], background_mechanism, foreground_mechanism
    )
    expected_contributions = unit_change_linear(
        background_mechanism, background_df, foreground_mechanism, foreground_df, ["A", "B"]
    )

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)

    background_mechanism = SklearnRegressionModel(RFR(n_jobs=1).fit(np.column_stack((A1, B1)), C1))
    foreground_mechanism = SklearnRegressionModel(RFR(n_jobs=1).fit(np.column_stack((A2, B2)), C2))

    actual_contributions = unit_change(
        background_df, foreground_df, ["A", "B"], background_mechanism, foreground_mechanism
    )
    expected_contributions = unit_change_nonlinear(
        background_mechanism,
        background_df,
        foreground_mechanism,
        foreground_df,
        ["A", "B"],
        shapley_config=ShapleyConfig(n_jobs=1),
    )

    np.testing.assert_array_almost_equal(actual_contributions, expected_contributions, decimal=1)


def test_given_reserved_column_name_f_when_calling_unit_change_linear_then_raises_value_error():
    bg_df = pd.DataFrame({"f": [1.0, 2.0], "x": [3.0, 4.0]})
    fg_df = pd.DataFrame({"f": [1.1, 2.1], "x": [3.1, 4.1]})
    mechanism = SklearnLinearRegressionModel(
        LinearRegression().fit(bg_df[["f", "x"]].to_numpy(), np.array([1.0, 2.0]))
    )
    with pytest.raises(ValueError, match="reserved"):
        unit_change_linear(mechanism, bg_df, mechanism, fg_df, ["f", "x"])


def test_given_reserved_column_name_f_when_calling_unit_change_nonlinear_then_raises_value_error():
    bg_df = pd.DataFrame({"f": [1.0, 2.0], "x": [3.0, 4.0]})
    fg_df = pd.DataFrame({"f": [1.1, 2.1], "x": [3.1, 4.1]})
    mechanism = SklearnRegressionModel(RFR(n_jobs=1).fit(bg_df[["f", "x"]].to_numpy(), np.array([1.0, 2.0])))
    with pytest.raises(ValueError, match="reserved"):
        unit_change_nonlinear(mechanism, bg_df, mechanism, fg_df, ["f", "x"])


def test_given_non_reserved_column_names_when_calling_unit_change_linear_then_returns_f_column_in_output():
    num_rows = 20
    rng = np.random.default_rng(42)
    A = rng.normal(size=num_rows)
    B = rng.normal(size=num_rows)
    C = 2 * A + 3 * B
    bg_df = pd.DataFrame({"A": A, "B": B})
    fg_df = pd.DataFrame({"A": A + 0.1, "B": B + 0.2})
    m = SklearnLinearRegressionModel(LinearRegression().fit(bg_df.to_numpy(), C))
    result = unit_change_linear(m, bg_df, m, fg_df, ["A", "B"])
    assert "f" in result.columns, "Mechanism column 'f' should be present in the output"
    assert "A" in result.columns
    assert "B" in result.columns
