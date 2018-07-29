import numpy as np
import pandas as pd
import pytest

from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand


@pytest.fixture(params=[(10, 5, 100000), (10, 5, 100000)])
def linear_dataset(request):  # beta, num_common_causes, num_samples):
    beta = request.param[0]
    num_common_causes = request.param[1]
    num_samples = request.param[2]
    range_c1 = beta * 0.5
    range_c2 = beta * 0.5
    means = np.random.uniform(-1, 1, num_common_causes)
    cov_mat = np.diag(np.ones(num_common_causes))
    X = np.random.multivariate_normal(means, cov_mat, num_samples)
    c1 = np.random.uniform(0, range_c1, num_common_causes)
    c2 = np.random.uniform(0, range_c2, num_common_causes)

    t = X @ c1  # + np.random.normal(0, 0.01)
    y = X @ c2 + beta * t  # + np.random.normal(0,0.01)
    print(c1)
    print(c2)
    ty = np.column_stack((t, y))
    data = np.column_stack((X, ty))

    treatment = "t"
    outcome = "y"
    common_causes = [("X" + str(i)) for i in range(0, num_common_causes)]
    ate = beta
    instruments = None
    other_variables = None
    col_names = common_causes + [treatment, outcome]
    data = pd.DataFrame(data,
                        columns=col_names)
    ret_dict = {
        "data": data,
        "treatment": treatment,
        "outcome": outcome,
        "common_causes": common_causes,
        "ate": beta
    }
    return ret_dict


# @pytest.mark.usefixtures("simulate_linear_dataset")
class TestLinearRegressionEstimator(object):

    @pytest.mark.parametrize("error_tolerance", [0.01, 0.05])
    def test_average_treatment_effect(self, linear_dataset, error_tolerance):
        data = linear_dataset["data"]
        true_ate = linear_dataset["ate"]
        target_estimand = IdentifiedEstimand(
            treatment_variable=linear_dataset["treatment"],
            outcome_variable=linear_dataset["outcome"],
            backdoor_variables=linear_dataset["common_causes"]
        )
        estimator_ate = LinearRegressionEstimator(
            data,
            identified_estimand=target_estimand,
            treatment=linear_dataset["treatment"],
            outcome=linear_dataset["outcome"]
        )

        est_ate = estimator_ate.estimate_effect()
        error = est_ate.value - true_ate
        print("Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
            error, error_tolerance * 100, est_ate.value, true_ate)
        )
        res = True if (error < true_ate * error_tolerance) else False
        assert res
