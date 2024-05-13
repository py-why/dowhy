import numpy as np, pandas as pd, networkx as nx
from dowhy.gcm.distribution_change_robust import distribution_change_robust
from dowhy.gcm.shapley import *
from dowhy import gcm
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from flaky import flaky
from pytest import approx

def _gen_data(seed=0, N=1000):    
    np.random.seed(seed)
    X1_old = np.random.normal(1,1,N)
    X2_old = 0.5*X1_old + np.random.normal(0,1,N)
    Y_old = X1_old + X2_old + 0.25*X1_old**2 + 0.25*X2_old**2 + np.random.normal(0,1,N)
    X1_new = np.random.normal(1,1.1,N)
    X2_new = 0.2*X1_new + np.random.normal(0,1,N)
    Y_new = X1_new + X2_new + 0.25*X1_new**2 - 0.25*X2_new**2 + np.random.normal(0,1,N)
    data_old = pd.DataFrame({'X1' : X1_old, 'X2' : X2_old, 'Y' : Y_old})
    data_new = pd.DataFrame({'X1' : X1_new, 'X2' : X2_new, 'Y' : Y_new})
    return data_old, data_new

def _true_theta(C):
    mX0, mX0sq = (1, 2.21) if C[0] else (1,2)
    mX1, mX1sq = (0.2*mX0,0.2**2*mX0sq+1) if C[1] else (0.5*mX0,0.5**2*mX0sq+1)
    mY = mX0 + mX1 + 0.25*mX0sq - 0.25*mX1sq if C[2] else mX0 + mX1 + 0.25*mX0sq + 0.25*mX1sq
    return mY

kwargs = {'regressor' : GradientBoostingRegressor, 'regressor_kwargs' : {'random_state' : 0},
          'classifier' : GradientBoostingClassifier, 'classifier_kwargs' : {'random_state' : 0},
          'calibrator' : LogisticRegression, 'calibrator_kwargs' : {'penalty' : None}, 
          'calib_size' : 0.2,
          'xfit' : True, 'xfit_folds' : 10,
         }

@flaky(max_runs=5)
def test_given_two_data_sets_with_different_mechanisms_when_evaluate_distribution_change_then_returns_expected_result():
    data_old, data_new = _gen_data()
    causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('X1', 'X2'), ('X1', 'Y'), ('X2', 'Y')]))
    gcm.auto.assign_causal_mechanisms(causal_model, data_old)
    shap = distribution_change_robust(causal_model, data_old, data_new, 'Y', **kwargs)

    true_shap = estimate_shapley_values(_true_theta, 3, ShapleyConfig(approximation_method = ShapleyApproximationMethods.EXACT))

    assert shap["X1"] == approx(true_shap[0], abs=0.1)
    assert shap["X2"] == approx(true_shap[1], abs=0.1)
    assert shap["Y"] == approx(true_shap[2], abs=0.1)