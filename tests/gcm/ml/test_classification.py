import numpy as np
from flaky import flaky

from dowhy.gcm.ml import create_polynom_logistic_regression_classifier


@flaky(max_runs=3)
def test_when_fit_and_predict_polynom_classifier_then_returns_accurate_results():
    def _generate_data():
        X = np.random.normal(0, 1, (1000, 2))
        Y = []

        for x in X:
            if x[0] * x[1] > 0:
                Y.append("Class 0")
            else:
                Y.append("Class 1")

        return X, np.array(Y)

    X_training, Y_training = _generate_data()
    X_test, Y_test = _generate_data()
    mdl = create_polynom_logistic_regression_classifier()
    mdl.fit(X_training, Y_training)

    assert np.sum(mdl.predict(X_test).reshape(-1) == Y_test) > 950
