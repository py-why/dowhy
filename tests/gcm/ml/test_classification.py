import numpy as np
from flaky import flaky

from dowhy.gcm.ml import create_hist_gradient_boost_classifier, create_polynom_logistic_regression_classifier
from dowhy.gcm.ml.classification import create_support_vector_classifier


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


@flaky(max_runs=3)
def test_given_categorical_training_data_with_many_categories_when_fit_classification_model_then_returns_reasonably_accurate_predictions():
    def _generate_data():
        X = np.column_stack(
            [np.random.choice(20, 1000, replace=True).astype(str), np.random.normal(0, 1, (1000, 2)).astype(object)]
        ).astype(object)
        Y = []

        for x in X:
            if int(x[0]) % 2 == 0:
                if x[1] * x[2] < 1:
                    Y.append("Class 0")
                elif x[1] * x[2] > 1:
                    Y.append("Class 1")
                else:
                    Y.append("Class 2")
            else:
                if x[1] + x[2] < 1:
                    Y.append("Class 2")
                elif x[1] + x[2] > 1:
                    Y.append("Class 1")
                else:
                    Y.append("Class 0")

        return X, np.array(Y)

    X_training, Y_training = _generate_data()
    X_test, Y_test = _generate_data()
    mdl = create_hist_gradient_boost_classifier()
    mdl.fit(X_training, Y_training)

    assert np.sum(mdl.predict(X_test).reshape(-1) == Y_test) > 950


def test_given_svc_model_then_supports_predict_probabilities():
    mdl = create_support_vector_classifier()
    mdl.fit(np.random.normal(0, 1, 100), np.random.choice(2, 100).astype(str))
    mdl.predict_probabilities(np.random.normal(0, 1, 10))
