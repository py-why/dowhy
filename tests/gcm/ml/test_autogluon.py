import tempfile

import numpy as np
from flaky import flaky
from pytest import approx, importorskip, mark
from sklearn.model_selection import train_test_split

from dowhy.gcm.causal_mechanisms import AdditiveNoiseModel, ClassifierFCM

autogluon = importorskip("dowhy.gcm.ml.autogluon")
from dowhy.gcm.ml.autogluon import AutoGluonClassifier, AutoGluonRegressor
from dowhy.gcm.util.general import shape_into_2d


@flaky(max_runs=3)
def test_given_linear_categorical_data_when_using_auto_gluon_classifier_linear_then_provides_good_fit():
    X, y = _generate_linear_classification_data(num_samples=1000, noise_std=0.2)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    y_test = shape_into_2d(y_test)
    with tempfile.TemporaryDirectory() as temp_dir:
        model = AutoGluonClassifier(path=temp_dir, time_limit=60, hyperparameters={"LR": {}})
        model.fit(x_train, y_train)

    assert np.sum(model.predict(x_test).squeeze() == y_test.squeeze()) / x_test.shape[0] == approx(1, abs=0.1)


@flaky(max_runs=3)
def test_given_linear_continuous_data_when_using_auto_gluon_regressor_linear_then_provides_good_fit():
    X, y, _ = _generate_linear_data(num_samples=2000, noise_std=0.2)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    y_test = shape_into_2d(y_test)
    with tempfile.TemporaryDirectory() as temp_dir:
        model = AutoGluonRegressor(path=temp_dir, time_limit=60, hyperparameters={"LR": {}})
        model.fit(x_train, y_train)

    assert np.std(model.predict(x_test) - y_test) == approx(0.2, abs=1)


def test_given_classifier_fcm_with_autogluon_classifier_model_when_drawing_samples_then_returns_samples():
    with tempfile.TemporaryDirectory() as temp_dir:
        autogluon_sem = ClassifierFCM(AutoGluonClassifier(path=temp_dir, time_limit=5))
        autogluon_sem.fit(np.random.normal(0, 1, (100, 3)), np.random.choice(3, 100, replace=True).astype(str))
    autogluon_sem.draw_samples(np.random.normal(0, 1, (5, 3)))


def test_given_anm_with_autogluon_regressor_model_when_drawing_samples_then_returns_samples():
    with tempfile.TemporaryDirectory() as temp_dir:
        autogluon_sem = AdditiveNoiseModel(AutoGluonClassifier(path=temp_dir, time_limit=5))
        autogluon_sem.fit(np.random.normal(0, 1, (100, 3)), np.random.normal(0, 1, (100, 1)))
    autogluon_sem.draw_samples(np.random.normal(0, 1, (5, 3)))


def _generate_linear_data(num_samples: int, noise_std: float):
    num_features = int(np.random.choice(20, 1) + 1)
    coefficients = np.random.uniform(-5, 5, num_features)

    X = np.random.normal(0, 1, (num_samples, num_features))
    y = np.sum(coefficients * X, axis=1) + np.random.normal(0, noise_std, num_samples)

    return X, y, coefficients


def _generate_linear_classification_data(num_samples: int, noise_std: float):
    X, y, _ = _generate_linear_data(num_samples, noise_std)

    y = y > np.median(y)

    return X, y.astype(str)
