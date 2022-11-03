import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm.unit_change import SklearnLinearRegressionModel, unit_change_linear, unit_change_nonlinear


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

    np.testing.assert_array_almost_equal(actual_contributions.to_numpy(), expected_contributions.to_numpy(), decimal=1)


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
        background_mechanism, background_df, foreground_mechanism, foreground_df, ["A", "B"]
    )

    np.testing.assert_array_almost_equal(actual_contributions.to_numpy(), expected_contributions.to_numpy(), decimal=1)


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
        )

    with pytest.raises(NotFittedError):
        unit_change_nonlinear(
            SklearnRegressionModel(RFR()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnRegressionModel(RFR()),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )


def test_given_fitted_nonlinnear_mechanisms_when_evaluate_unit_change_linear_method_then_raises_exception():
    with pytest.raises(AttributeError):
        unit_change_linear(
            SklearnRegressionModel(RFR().fit(np.random.normal(size=(100, 2)), np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            SklearnRegressionModel(RFR().fit(np.random.normal(size=(100, 2)), np.random.normal(size=100))),
            pd.DataFrame(data=dict(A=np.random.normal(size=100), B=np.random.normal(size=100))),
            ["A", "B"],
        )
