import itertools
import sys

import numpy as np
import pandas as pd
import pytest

import dowhy.datasets
from dowhy import EstimandType, identify_effect_auto
from dowhy.graph import build_graph_from_str

from .example_graphs import TEST_GRAPHS


class SimpleEstimator(object):
    def __init__(self, error_tolerance, Estimator, identifier_method="backdoor"):
        print("Error tolerance is", error_tolerance)
        self._error_tolerance = error_tolerance
        self._Estimator = Estimator
        self._identifier_method = identifier_method

    def average_treatment_effect_test(
        self,
        dataset="linear",
        beta=10,
        num_common_causes=1,
        num_instruments=1,
        num_effect_modifiers=0,
        num_treatments=1,
        num_frontdoor_variables=0,
        num_samples=100000,
        treatment_is_binary=True,
        treatment_is_category=False,
        outcome_is_binary=False,
        confidence_intervals=False,
        test_significance=False,
        method_params=None,
    ):

        # generalized adjustment identification requires python >=3.10
        if sys.version_info < (3, 10) and self._identifier_method == "general_adjustment":
            return

        if dataset == "linear":
            data = dowhy.datasets.linear_dataset(
                beta=beta,
                num_common_causes=num_common_causes,
                num_instruments=num_instruments,
                num_effect_modifiers=num_effect_modifiers,
                num_treatments=num_treatments,
                num_frontdoor_variables=num_frontdoor_variables,
                num_samples=num_samples,
                treatment_is_binary=treatment_is_binary,
                treatment_is_category=treatment_is_category,
                outcome_is_binary=outcome_is_binary,
            )
        elif dataset == "simple-iv":
            data = dowhy.datasets.simple_iv_dataset(
                beta=beta,
                num_treatments=num_treatments,
                num_samples=num_samples,
                treatment_is_binary=treatment_is_binary,
                outcome_is_binary=outcome_is_binary,
            )
        else:
            raise ValueError("Dataset type not supported.")

        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method(self._identifier_method)
        estimator_ate = self._Estimator(
            identified_estimand=target_estimand,
            **method_params,
        )

        estimator_ate.fit(
            data["df"],
            effect_modifier_names=data["effect_modifier_names"],
        )

        true_ate = data["ate"]
        ate_estimate = estimator_ate.estimate_effect(
            data["df"],
            control_value=0,
            treatment_value=1,
            test_significance=test_significance,
            evaluate_effect_strength=False,
            confidence_intervals=confidence_intervals,
            target_units="ate",
            **method_params,
        )
        str(ate_estimate)  # checking if str output is correctly created
        error = abs(ate_estimate.value - true_ate)
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, self._error_tolerance * 100, ate_estimate.value, true_ate
            )
        )
        res = True if (error < abs(true_ate) * self._error_tolerance) else False
        assert res
        # Compute confidence intervals, standard error and significance tests
        if confidence_intervals:
            ate_estimate.get_confidence_intervals()
            ate_estimate.get_confidence_intervals(confidence_level=0.99)
            ate_estimate.get_confidence_intervals(method="bootstrap")
            ate_estimate.get_standard_error()
            ate_estimate.get_standard_error(method="bootstrap")
        if test_significance:
            ate_estimate.test_stat_significance()
            ate_estimate.test_stat_significance(method="bootstrap")

    def average_treatment_effect_testsuite(
        self,
        tests_to_run="all",
        num_common_causes=[2, 3],
        num_instruments=[
            1,
        ],
        num_effect_modifiers=[
            0,
        ],
        num_treatments=[
            1,
        ],
        num_frontdoor_variables=[
            0,
        ],
        treatment_is_binary=[
            True,
        ],
        treatment_is_category=[
            False,
        ],
        outcome_is_binary=[
            False,
        ],
        confidence_intervals=[
            False,
        ],
        test_significance=[
            False,
        ],
        dataset="linear",
        method_params=None,
    ):

        # generalized adjustment identification requires python >=3.10
        if sys.version_info < (3, 10) and self._identifier_method == "general_adjustment":
            return

        args_dict = {
            "num_common_causes": num_common_causes,
            "num_instruments": num_instruments,
            "num_effect_modifiers": num_effect_modifiers,
            "num_treatments": num_treatments,
            "num_frontdoor_variables": num_frontdoor_variables,
            "treatment_is_binary": treatment_is_binary,
            "treatment_is_category": treatment_is_category,
            "outcome_is_binary": outcome_is_binary,
            "confidence_intervals": confidence_intervals,
            "test_significance": test_significance,
        }
        keys, values = zip(*args_dict.items())
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for cfg in configs:
            print("\nConfig:", cfg)
            cfg["dataset"] = dataset
            cfg["method_params"] = method_params
            self.average_treatment_effect_test(**cfg)

    def custom_data_average_treatment_effect_test(self, data, method_params={}):
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )

        # generalized adjustment identification requires python >=3.10
        if sys.version_info < (3, 10) and self._identifier_method == "general_adjustment":
            return

        target_estimand.set_identifier_method(self._identifier_method)
        estimator_ate = self._Estimator(identified_estimand=target_estimand, test_significance=None, **method_params)
        estimator_ate.fit(data["df"])
        true_ate = data["ate"]
        ate_estimate = estimator_ate.estimate_effect(data["df"])
        error = abs(ate_estimate.value - true_ate)
        print(
            "Error in ATE estimate = {0} with tolerance {1}%. Estimated={2},True={3}".format(
                error, self._error_tolerance * 100, ate_estimate.value, true_ate
            )
        )
        res = True if (error < abs(true_ate) * self._error_tolerance) else False
        assert res


class SimpleEstimatorWithModelParams(object):
    def __init__(self, Estimator, method_params, identifier_method="backdoor"):
        self._Estimator = Estimator
        self._method_params = method_params
        self._identifier_method = identifier_method

    def consistent_estimator_encoding_test(self):
        """
        This test tries to verify and enforce consistent encoding of categorical variables
        by Estimators. The desired behaviour is that encodings of new values are produced
        during `fit()`, which is also when the model is learned/trained/fitted.

        In `estimator.estimate_effect()` and `do(x)` the same encodings should be reused.
        """

        # Generate a dataset with some categorical variables (common causes)
        # This configuration is necessary for the test and should not be varied.
        data = dowhy.datasets.linear_dataset(
            beta=1,
            num_common_causes=3,
            num_discrete_common_causes=2,
            num_instruments=2,
            num_effect_modifiers=0,
            num_discrete_effect_modifiers=0,
            num_treatments=1,
            num_frontdoor_variables=0,
            num_samples=500,
            treatment_is_binary=True,
            treatment_is_category=False,
            outcome_is_binary=False,
        )

        # For the purposes of the test, these are the categorical columns.
        encoded_categorical_columns = ["W1", "W2"]

        # Since their values are integer, convert them to string type to ensure
        # Categorical handling.
        df_1 = data["df"]
        df_1[encoded_categorical_columns] = df_1[encoded_categorical_columns].astype(str)

        def fit_estimator(data, method_params):
            """
            Creates an Estimator, identifies the effect, and fits the Estimator to the data.
            The Estimator is returned.
            """

            target_estimand = identify_effect_auto(
                build_graph_from_str(data["gml_graph"]),
                observed_nodes=list(data["df"].columns),
                action_nodes=data["treatment_name"],
                outcome_nodes=data["outcome_name"],
                estimand_type=EstimandType.NONPARAMETRIC_ATE,
            )
            target_estimand.set_identifier_method(self._identifier_method)

            estimator = self._Estimator(
                identified_estimand=target_estimand,
                **method_params,
            )

            estimator.fit(
                df_1,
                effect_modifier_names=data["effect_modifier_names"],
            )
            return estimator

        def estimate(estimator, df):
            """
            Returns an Estimate of the ATE, given the data provided.
            """
            print(f"Est,,, {type(estimator)}")
            estimate = estimator.estimate_effect(
                df,
                control_value=0,
                treatment_value=1,
                test_significance=False,
                evaluate_effect_strength=False,
                confidence_intervals=False,
                target_units="ate",
            )
            return estimate

        def swap_first_row(df: pd.DataFrame, columns: list):
            """
            A property of some categorical encoders (e.g. Pandas' `get_dummies()`) is that the
            values of encoded variables are assigned on the order in which each unique value is
            encountered in the data. Therefore, by swapping the *first* row of some data with
            some other row, we can try to try the encoder into a different encoding of the data.

            This function finds a row which is dissimilar to the first row in terms of all the
            values in `columns`. This row is then swapped with row 0. A copy of the data with the
            rows swapped is returned.

            :param df: A DataFrame.
            :param columns: A list of column names, the values of which must be dissimilar to the first row
            :returns: A copy of df in which the first row is swapped with another row.
            """
            # Get the values of row 0 for the specified columns
            row_0_values = df.loc[0, columns].tolist()

            # Find rows where values differ from row 0 in terms of all values in the specified columns
            n = df[(df[columns] != row_0_values).all(axis=1)].index[0]

            # Create a copy of the data and swap the rows.
            df_swap = df.copy()
            df_swap.iloc[0] = df.iloc[n]
            df_swap.iloc[n] = df.iloc[0]
            return df_swap

        # Test 1: Permuting data order does not affect Effect estimate.
        # This test will not likely fail with a RegressionEstimator, because
        # the effect of common cause variables is additive and does not contribute to
        # the estimated effect. However, it could fail with other Estimators.
        estimator = fit_estimator(data, self._method_params)
        estimate_1 = estimate(estimator, df_1)
        df_2 = swap_first_row(df_1, encoded_categorical_columns)
        estimate_2 = estimate(estimator, df_2)

        error = abs(estimate_1.value - estimate_2.value)
        error_tolerance = 1.0e-6  # tiny errors OK due to e.g. precision errors
        print(
            "Difference {0} between ATE estimates 1: {1} and 2: {2} must be < {3}".format(
                error,
                estimate_1.value,
                estimate_2.value,
                error_tolerance,
            )
        )
        assert error < error_tolerance

        # Test 2: Verify that estimated Outcomes from "do-operator" are unchanged
        # While for some Estimators the Effect is unaffected by changes to common-cause
        # data, and data ordering, predicted Outcomes should be as all variables can
        # contribute to these Outcomes. However, they are only available in a standard
        # interface for Estimators which support `do(x)`.
        #
        # In this test, we verify that the result of `do(x)` does not change when we
        # present our two datasets (one has swapped first row). If the result differs,
        # this is likely due to the encoding of these new data, since the model is
        # unchanged.
        #
        # Unlike the Effect test #1 above, this test is verifiable; we can randomize
        # the values and combinations of the encoded values and verify that under these
        # conditions the result of `do(x)` *does* change. This is the type of error we
        # expect to observe if there's an encoding error - all the encoded variables
        # would change.
        def randomize_column_values(df, columns):
            """
            Returns a copy of `df` with randomized values for specified `columns`.
            Randomized values are chosen uniformly from the set of unique values
            in each specified column. This action should disrupt the result of `do(x)`
            on the data.
            """
            df = df.copy()
            num_rows = len(df)
            for column in columns:
                possible_values = df[column].unique()
                df[column] = np.random.choice(possible_values, num_rows)
            return df

        try:
            df_3 = randomize_column_values(df_1, encoded_categorical_columns)

            treatment_value = 1
            do_x_with_df_1 = estimator.do(x=treatment_value, data_df=df_1)
            do_x_with_df_2 = estimator.do(x=treatment_value, data_df=df_2)
            do_x_with_df_3 = estimator.do(x=treatment_value, data_df=df_3)

            # Test that do(x) result is unchanged despite row permutation
            error_2 = abs(do_x_with_df_1 - do_x_with_df_2)
            print(
                "Difference {0} between do(x) 1: {1} and 2: {2} must be < {3}".format(
                    error_2,
                    do_x_with_df_1,
                    do_x_with_df_2,
                    error_tolerance,
                )
            )
            assert error_2 < error_tolerance

            # Verify that this test *does* detect errors, when common-cause data changed.
            error_3 = abs(do_x_with_df_1 - do_x_with_df_3)
            print(
                "Difference {0} between do(x) 1: {1} and 3: {2} must be > {3}".format(
                    error_3,
                    do_x_with_df_1,
                    do_x_with_df_3,
                    error_tolerance,
                )
            )
            assert error_3 > error_tolerance
        except NotImplementedError:
            pass  # Expected, for many Estimators


class TestGraphObject(object):
    def __init__(
        self,
        graph_str,
        observed_variables,
        action_nodes,
        outcome_node,
    ):
        self.graph = build_graph_from_str(graph_str)
        self.action_nodes = action_nodes
        self.outcome_node = outcome_node
        self.observed_nodes = observed_variables


@pytest.fixture(params=TEST_GRAPHS.keys())
def example_graph(request):
    return TestGraphObject(**TEST_GRAPHS[request.param])
