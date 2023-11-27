import numpy as np
import pandas as pd
from pytest import mark
from sklearn.linear_model import LinearRegression

import dowhy.api
import dowhy.datasets


@mark.usefixtures("fixed_seed")
class TestPandasDoAPI(object):
    @mark.parametrize(
        ["N", "error_tolerance"],
        [
            (10000, 0.1),
        ],
    )
    def test_pandas_api_discrete_cause_continuous_confounder(self, N, error_tolerance):
        data = dowhy.datasets.linear_dataset(
            beta=10, num_common_causes=1, num_instruments=1, num_samples=N, treatment_is_binary=False
        )
        X0 = np.random.normal(size=N)
        v = (np.random.normal(size=N) + X0).astype(int)
        y = data["ate"] * v + X0 + np.random.normal()
        data["df"]["v"] = v
        data["df"]["X0"] = X0
        data["df"]["y"] = y
        df = data["df"].copy()

        variable_types = {"v": "d", "X0": "c", "y": "c"}
        outcome = "y"
        cause = "v"
        common_causes = "X0"
        method = "weighting"
        causal_df = df.causal.do(
            x=cause,
            variable_types=variable_types,
            outcome=outcome,
            method=method,
            common_causes=common_causes,
        )

        ate = (causal_df[causal_df.v == 1].mean() - causal_df[causal_df.v == 0].mean())["y"]
        error = np.abs(ate - data["ate"])
        res = True if (error < data["ate"] * error_tolerance) else False
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, error_tolerance * 100, ate, data["ate"]
            )
        )
        assert res

    @mark.parametrize(
        ["N", "error_tolerance"],
        [
            (10000, 0.1),
        ],
    )
    def test_pandas_api_discrete_cause_discrete_confounder(self, N, error_tolerance):
        data = dowhy.datasets.linear_dataset(
            beta=10, num_common_causes=1, num_instruments=1, num_samples=N, treatment_is_binary=False
        )
        X0 = np.random.normal(size=N).astype(int)
        v = (np.random.normal(size=N) + X0).astype(int)
        y = data["ate"] * v + X0 + np.random.normal()
        data["df"]["v"] = v
        data["df"]["X0"] = X0
        data["df"]["y"] = y
        df = data["df"].copy()

        variable_types = {"v": "d", "X0": "d", "y": "c"}
        outcome = "y"
        cause = "v"
        common_causes = "X0"
        method = "weighting"
        causal_df = df.causal.do(
            x=cause,
            variable_types=variable_types,
            outcome=outcome,
            method=method,
            common_causes=common_causes,
        )

        ate = (causal_df[causal_df.v == 1].mean() - causal_df[causal_df.v == 0].mean())["y"]
        print("ate", ate)
        error = np.abs(ate - data["ate"])
        res = True if (error < data["ate"] * error_tolerance) else False
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, error_tolerance * 100, ate, data["ate"]
            )
        )
        assert res

    @mark.parametrize(
        ["N", "error_tolerance"],
        [
            (10000, 0.1),
        ],
    )
    def test_pandas_api_continuous_cause_discrete_confounder(self, N, error_tolerance):
        data = dowhy.datasets.linear_dataset(
            beta=10, num_common_causes=1, num_instruments=1, num_samples=N, treatment_is_binary=False
        )
        X0 = np.random.normal(size=N).astype(int)
        v = np.random.normal(size=N) + X0
        y = data["ate"] * v + X0 + np.random.normal()
        data["df"]["v"] = v
        data["df"]["X0"] = X0
        data["df"]["y"] = y
        df = data["df"].copy()

        variable_types = {"v": "c", "X0": "d", "y": "c"}
        outcome = "y"
        cause = "v"
        common_causes = "X0"
        method = "weighting"
        causal_df = df.causal.do(
            x=cause,
            variable_types=variable_types,
            outcome=outcome,
            method=method,
            common_causes=common_causes,
        )

        ate = LinearRegression().fit(causal_df[["v"]], causal_df["y"]).coef_[0]
        print("ate", ate)
        error = np.abs(ate - data["ate"])
        res = True if (error < data["ate"] * error_tolerance) else False
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, error_tolerance * 100, ate, data["ate"]
            )
        )
        assert res

    @mark.parametrize(
        ["N", "error_tolerance"],
        [
            (10000, 0.1),
        ],
    )
    def test_pandas_api_continuous_cause_continuous_confounder(self, N, error_tolerance):
        data = dowhy.datasets.linear_dataset(
            beta=10, num_common_causes=1, num_instruments=1, num_samples=N, treatment_is_binary=False
        )
        X0 = np.random.normal(size=N)
        v = np.random.normal(size=N) + X0
        y = data["ate"] * v + X0 + np.random.normal()
        data["df"]["v"] = v
        data["df"]["X0"] = X0
        data["df"]["y"] = y
        df = data["df"].copy()

        variable_types = {"v": "c", "X0": "c", "y": "c"}
        outcome = "y"
        cause = "v"
        common_causes = "X0"
        method = "weighting"
        causal_df = df.causal.do(
            x=cause,
            variable_types=variable_types,
            outcome=outcome,
            method=method,
            common_causes=common_causes,
        )

        ate = LinearRegression().fit(causal_df[["v"]], causal_df["y"]).coef_[0]
        print("ate", ate)
        error = np.abs(ate - data["ate"])
        res = True if (error < data["ate"] * error_tolerance) else False
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, error_tolerance * 100, ate, data["ate"]
            )
        )
        assert res

    """
    In the following three tests, we have made use of the assert True at the end, but it is not
    a tautology due to the fact the function being tested has the ability to raise an exception
    when it belives that the behavior of the function is wrong.
    """

    @mark.parametrize(
        ["N", "variable_types"],
        [
            (10000, {"v0": "b", "y": "c", "W0": "c"}),
        ],
    )
    def test_pandas_api_with_full_specification_of_type(self, N, variable_types):
        data = dowhy.datasets.linear_dataset(
            beta=5, num_common_causes=1, num_instruments=0, num_samples=1000, treatment_is_binary=True
        )

        data["df"].causal.do(x="v0", variable_types=variable_types, outcome="y", common_causes=["W0"]).groupby(
            "v0"
        ).mean()
        assert True

    @mark.parametrize(
        ["N", "variable_types"],
        [
            (10000, {"v0": "b", "W0": "c"}),
        ],
    )
    def test_pandas_api_with_partial_specification_of_type(self, N, variable_types):
        data = dowhy.datasets.linear_dataset(
            beta=5, num_common_causes=1, num_instruments=0, num_samples=1000, treatment_is_binary=True
        )

        data["df"].causal.do(x="v0", variable_types=variable_types, outcome="y", common_causes=["W0"]).groupby(
            "v0"
        ).mean()
        assert True

    @mark.parametrize(
        ["N", "variable_types"],
        [
            (10000, {}),
        ],
    )
    def test_pandas_api_with_no_specification_of_type(self, N, variable_types):
        data = dowhy.datasets.linear_dataset(
            beta=5, num_common_causes=1, num_instruments=0, num_samples=1000, treatment_is_binary=True
        )

        data["df"].causal.do(x="v0", variable_types=variable_types, outcome="y", common_causes=["W0"]).groupby(
            "v0"
        ).mean()
        assert True

    @mark.parametrize(
        ["N", "variable_types"],
        [
            (1, {"v0": "b", "W0": "c"}),
        ],
    )
    def test_pandas_api_with_dummy_data(self, N, variable_types):
        df = pd.DataFrame({"x": [0, 0.5, 1], "y": [1, 0.5, 0], "a": [0, 0.5, 0], "b": [0.25, 0, 0]})
        dd = df.causal.do(
            x=["x"],
            outcome="y",
            common_causes=["a", "b"],
            variable_types=dict(x="c", y="c", a="c", b="c"),
        )
        print(dd)
