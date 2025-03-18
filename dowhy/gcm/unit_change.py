"""This module provides the APIs for attributing the change in the output value of a deterministic mechanism for a statistical unit."""

from abc import abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted

from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values


class LinearPredictionModel:
    @property
    @abstractmethod
    def coefficients(self) -> np.ndarray:
        pass


class SklearnLinearRegressionModel(SklearnRegressionModel, LinearPredictionModel):
    def __init__(self, sklearn_mdl: LinearModel) -> None:
        super(SklearnLinearRegressionModel, self).__init__(sklearn_mdl)

    @property
    def coefficients(self) -> np.ndarray:
        check_is_fitted(self.sklearn_model)
        return self.sklearn_model.coef_


def unit_change(
    background_df: pd.DataFrame,
    foreground_df: pd.DataFrame,
    input_column_names: List[str],
    background_mechanism: PredictionModel,
    foreground_mechanism: Optional[PredictionModel] = None,
    shapley_config: Optional[ShapleyConfig] = None,
) -> pd.DataFrame:
    """
    This function attributes the change in the output value of a deterministic mechanism for a statistical unit to each input and optionally for the mechanism if `foreground_mechanism` is provided.
    The technical method is described in the following research paper:
    Kailash Budhathoki, George Michailidis, Dominik Janzing. *Explaining the root causes of unit-level changes*. arXiv, 2022.

    :param background_df: The background dataset.
    :param foreground_df: The foreground dataset.
    :param input_column_names: The names of the input columns.
    :param background_mechanism: The background mechanism. If the mechanism does not change, then this mechanism is used for attribution.
    :param foreground_mechanism: The foreground mechanism. If provided, the method also attributes the output change to the change in the mechanism.
    :param shapley_config: The configuration for calculating Shapley values.
    :return: A dataframe containing the contributions of each input and optionally the mechanism to the change in the output values of the deterministic mechanism(s) for given inputs.
    """
    if foreground_mechanism:
        if isinstance(background_mechanism, LinearPredictionModel):
            return unit_change_linear(
                background_mechanism, background_df, foreground_mechanism, foreground_df, input_column_names
            )
        else:
            return unit_change_nonlinear(
                background_mechanism,
                background_df,
                foreground_mechanism,
                foreground_df,
                input_column_names,
                shapley_config,
            )

    if isinstance(background_mechanism, LinearPredictionModel):
        return unit_change_linear_input_only(background_mechanism, background_df, foreground_df, input_column_names)
    else:
        return unit_change_nonlinear_input_only(
            background_mechanism, background_df, foreground_df, input_column_names, shapley_config
        )


def unit_change_nonlinear(
    background_mechanism: PredictionModel,
    background_df: pd.DataFrame,
    foreground_mechanism: PredictionModel,
    foreground_df: pd.DataFrame,
    input_column_names: List[str],
    shapley_config: Optional[ShapleyConfig] = None,
) -> pd.DataFrame:
    """
    Calculates the contributions of mechanism and each input to the change in the output values of a non-linear deterministic mechanism.
    The technical method is described in the following research paper:
    Kailash Budhathoki, George Michailidis, Dominik Janzing. *Explaining the root causes of unit-level changes*. arXiv, 2022.

    :param background_mechanism: The background mechanism.
    :param background_df: The background data.
    :param foreground_mechanism: The foreground mechanism.
    :param foreground_df: The foreground data.
    :param input_column_names: The names of the input (features) columns in both dataframes.
    :param shapley_config: The configuration for calculating Shapley values.
    :return: A pandas dataframe with attributions to each cause for the change in each output row of provided dataframes.
    """
    _check_if_input_columns_exist(background_df, foreground_df, input_column_names)

    def payoff(player_indicator: List[int]) -> np.ndarray:
        """The last cell in the binary vector represents the player 'mechanism'."""
        input_arrays = []
        for i, is_player_active in enumerate(player_indicator[:-1]):
            selected_df = foreground_df if is_player_active else background_df
            input_arrays.append(selected_df[input_column_names[i]].to_numpy())
        mechanism = foreground_mechanism if player_indicator[-1] else background_mechanism
        return mechanism.predict(np.column_stack(input_arrays)).flatten()

    contributions = estimate_shapley_values(payoff, len(input_column_names) + 1, shapley_config)
    root_causes = input_column_names + ["f"]
    return pd.DataFrame(contributions, columns=root_causes)


def unit_change_linear(
    background_mechanism: LinearPredictionModel,
    background_df: pd.DataFrame,
    foreground_mechanism: LinearPredictionModel,
    foreground_df: pd.DataFrame,
    input_column_names: List[str],
) -> pd.DataFrame:
    """
    Calculates the contributions of mechanism and each input to the change in the output values of a linear deterministic mechanism.

    :param background_mechanism: The linear background mechanism.
    :param background_df: The background data.
    :param foreground_mechanism: The linear foreground mechanism.
    :param foreground_df: The foreground data.
    :param input_column_names: The names of the input columns in both dataframes.
    :return: A pandas dataframe with attributions to each cause for the change in each output row of provided dataframes.
    """
    _check_if_input_columns_exist(background_df, foreground_df, input_column_names)

    coeffs_total = background_mechanism.coefficients + foreground_mechanism.coefficients  # p x 1
    coeffs_diff = foreground_mechanism.coefficients - background_mechanism.coefficients  # p x 1

    input_total = foreground_df[input_column_names].to_numpy() + background_df[input_column_names].to_numpy()  # n x p
    input_diff = foreground_df[input_column_names].to_numpy() - background_df[input_column_names].to_numpy()  # n x p

    contribution_input = 0.5 * np.einsum("ij,ki->ki", coeffs_total.reshape(-1, 1), input_diff)
    contribution_mechanism = 0.5 * np.einsum("ij,ki->k", coeffs_diff.reshape(-1, 1), input_total)
    contribution_df = pd.DataFrame(contribution_input, columns=input_column_names)
    contribution_df["f"] = contribution_mechanism  # TODO: Handle the case where 'f' is an input column name
    return contribution_df


def unit_change_nonlinear_input_only(
    mechanism: PredictionModel,
    background_df: pd.DataFrame,
    foreground_df: pd.DataFrame,
    input_column_names: List[str],
    shapley_config: Optional[ShapleyConfig] = None,
) -> pd.DataFrame:
    """
    Calculates the contributions of each input to the change in the output values of a non-linear deterministic mechanism.
    The technical method is a modification of the attribution method described in the following research paper, without mechanism as a player:
    Kailash Budhathoki, George Michailidis, Dominik Janzing. *Explaining the root causes of unit-level changes*. arXiv, 2022.

    :param mechanism: The mechanism.
    :param background_df: The background data.
    :param foreground_df: The foreground data.
    :param input_column_names: The names of the input (features) columns in both dataframes.
    :param shapley_config: The configuration for calculating Shapley values.
    :return: A pandas dataframe with attributions to each cause for the change in each output row of provided dataframes.
    """
    _check_if_input_columns_exist(background_df, foreground_df, input_column_names)

    def payoff(player_indicator: List[int]) -> np.ndarray:
        input_arrays = []
        for i, is_player_active in enumerate(player_indicator):
            selected_df = foreground_df if is_player_active else background_df
            input_arrays.append(selected_df[input_column_names[i]].to_numpy())
        return mechanism.predict(np.column_stack(input_arrays)).flatten()

    contributions = estimate_shapley_values(payoff, len(input_column_names), shapley_config)
    return pd.DataFrame(contributions, columns=input_column_names)


def unit_change_linear_input_only(
    mechanism: LinearPredictionModel,
    background_df: pd.DataFrame,
    foreground_df: pd.DataFrame,
    input_column_names: List[str],
) -> pd.DataFrame:
    """
    Calculates the contributions of each input to the change in the output values of a linear deterministic mechanism.

    :param mechanism: The linear mechanism.
    :param background_df: The background data.
    :param foreground_df: The foreground data.
    :param input_column_names: The names of the input (features) columns in both dataframes.
    :return: A pandas dataframe with attributions to each cause for the change in each output row of provided dataframes.
    """
    _check_if_input_columns_exist(background_df, foreground_df, input_column_names)

    input_diff = foreground_df[input_column_names].to_numpy() - background_df[input_column_names].to_numpy()  # n x p
    contribution_input = np.einsum("ij,ki->ki", mechanism.coefficients.reshape(-1, 1), input_diff)
    return pd.DataFrame(contribution_input, columns=input_column_names)


def _check_if_input_columns_exist(
    background_df: pd.DataFrame, foreground_df: pd.DataFrame, input_column_names: List[str]
) -> None:
    if not len(set(background_df.columns).intersection(input_column_names)) == len(input_column_names) or not len(
        set(foreground_df.columns).intersection(input_column_names)
    ) == len(input_column_names):
        raise ValueError("Input column names not found in either the background or the foreground data.")
