import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy import gcm


def _gen_data(N=10000):
    X1_old = np.random.normal(1, 1, N)
    X2_old = 0.5 * X1_old + np.random.normal(0, 1, N)
    Y_old = X1_old + X2_old + 0.25 * X1_old**2 + 0.25 * X2_old**2 + np.random.normal(0, 1, N)
    X1_new = np.random.normal(1, 1.1, N)
    X2_new = 0.2 * X1_new + np.random.normal(0, 1, N)
    Y_new = X1_new + X2_new + 0.25 * X1_new**2 - 0.25 * X2_new**2 + np.random.normal(0, 1, N)
    data_old = pd.DataFrame({"X1": X1_old, "X2": X2_old, "Y": Y_old})
    data_new = pd.DataFrame({"X1": X1_new, "X2": X2_new, "Y": Y_new})
    return data_old, data_new


@flaky(max_runs=5)
def test_given_two_data_sets_with_different_mechanisms_when_evaluate_distribution_change_then_returns_expected_result_mean():
    data_old, data_new = _gen_data()
    causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X1", "Y"), ("X2", "Y")]))
    shap = gcm.distribution_change_robust(
        causal_model,
        data_old,
        data_new,
        "Y",
        regressor=gcm.ml.regression.create_knn_regressor,
        xfit_folds=10,
    )

    assert shap["X1"] == approx(0.054, abs=0.1)
    assert shap["X2"] == approx(-0.298, abs=0.1)
    assert shap["Y"] == approx(-0.651, abs=0.1)


@flaky(max_runs=5)
def test_given_two_data_sets_with_different_mechanisms_when_evaluate_distribution_change_then_returns_expected_result_variance():
    data_old, data_new = _gen_data()
    causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X1", "Y"), ("X2", "Y")]))
    shap = gcm.distribution_change_robust(
        causal_model,
        data_old,
        data_new,
        "Y",
        regressor=gcm.ml.regression.create_knn_regressor,
        target_functional="variance",
        xfit_folds=10,
    )

    assert shap["X1"] == approx(0.811, abs=0.2)
    assert shap["X2"] == approx(-1.343, abs=0.2)
    assert shap["Y"] == approx(-1.397, abs=0.2)
