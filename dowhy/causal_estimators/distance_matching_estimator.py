from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dowhy.causal_estimator import CausalEstimate, CausalEstimator


class DistanceMatchingEstimator(CausalEstimator):
    """Simple matching estimator for binary treatments based on a distance
    metric.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """

    # allowed types of distance metric
    Valid_Dist_Metric_Params = ["p", "V", "VI", "w"]

    def __init__(
        self,
        identified_estimand,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        num_null_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations=CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction=CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level=CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates="auto",
        num_quantiles_to_discretize_cont_cols=CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        num_matches_per_unit=1,
        distance_metric="minkowski",
        exact_match_cols=None,
        **kwargs,
    ):
        """
        :param num_matches_per_unit: The number of matches per data point.
            Default=1.
        :param distance_metric: Distance metric to use. Default="minkowski"
            that corresponds to Euclidean distance metric with p=2.
        :param exact_match_cols: List of column names whose values should be
        exactly matched. Typically used for columns with discrete values.

        """
        # Required to ensure that self.method_params contains all the
        # parameters to create an object of this class
        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            num_matches_per_unit=num_matches_per_unit,
            distance_metric=distance_metric,
            exact_match_cols=exact_match_cols,
            **kwargs,
        )
        # Check if the treatment is one-dimensional
        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        # Checking if the treatment is binary
        if not pd.api.types.is_bool_dtype(self._data[self._treatment_name[0]]):
            error_msg = "Distance Matching method is applicable only for binary treatments"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        self.num_matches_per_unit = num_matches_per_unit
        self.distance_metric = distance_metric
        self.exact_match_cols = exact_match_cols

        # Dictionary of any user-provided params for the distance metric
        # that will be passed to sklearn nearestneighbors
        self.distance_metric_params = {}
        for param_name in self.Valid_Dist_Metric_Params:
            param_val = getattr(self, param_name, None)
            if param_val is not None:
                self.distance_metric_params[param_name] = param_val

        self.logger.info("INFO: Using Distance Matching Estimator")

        self.matched_indices_att = None
        self.matched_indices_atc = None

    def fit(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        exact_match_cols=None,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        self.set_data(data, treatment_name, outcome_name)
        self.exact_match_cols = exact_match_cols

        self.set_effect_modifiers(effect_modifier_names)

        self.logger.debug("Back-door variables used:" + ",".join(self._target_estimand.get_backdoor_variables()))

        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if self._observed_common_causes_names:
            if self.exact_match_cols is not None:
                self._observed_common_causes_names = [
                    v for v in self._observed_common_causes_names if v not in self.exact_match_cols
                ]
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            # Convert the categorical variables into dummy/indicator variables
            # Basically, this gives a one hot encoding for each category
            # The first category is taken to be the base line.
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = None
            error_msg = "No common causes/confounders present. Distance matching methods are not applicable"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        return self

    def estimate_effect(self, treatment_value: Any = 1, control_value: Any = 0, target_units=None, **_):
        # this assumes a binary treatment regime
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        updated_df = pd.concat(
            [self._observed_common_causes, self._data[[self._outcome_name, self._treatment_name[0]]]], axis=1
        )
        if self.exact_match_cols is not None:
            updated_df = pd.concat([updated_df, self._data[self.exact_match_cols]], axis=1)
        treated = updated_df.loc[self._data[self._treatment_name[0]] == 1]
        control = updated_df.loc[self._data[self._treatment_name[0]] == 0]
        numtreatedunits = treated.shape[0]
        numcontrolunits = control.shape[0]

        fit_att, fit_atc = False, False
        est = None
        # TODO remove neighbors that are more than a given radius apart
        if target_units == "att":
            fit_att = True
        elif target_units == "atc":
            fit_atc = True
        elif target_units == "ate":
            fit_att = True
            fit_atc = True
        else:
            raise ValueError("Target units string value not supported")

        if fit_att:
            # estimate ATT on treated by summing over difference between matched neighbors
            if self.exact_match_cols is None:
                control_neighbors = NearestNeighbors(
                    n_neighbors=self.num_matches_per_unit,
                    metric=self.distance_metric,
                    algorithm="ball_tree",
                    **self.distance_metric_params,
                ).fit(control[self._observed_common_causes.columns].values)
                distances, indices = control_neighbors.kneighbors(treated[self._observed_common_causes.columns].values)
                self.logger.debug("distances:")
                self.logger.debug(distances)

                att = 0

                for i in range(numtreatedunits):
                    treated_outcome = treated.iloc[i][self._outcome_name].item()
                    control_outcome = np.mean(control.iloc[indices[i]][self._outcome_name].values)
                    att += treated_outcome - control_outcome

                att /= numtreatedunits
                if target_units == "att":
                    est = att
                elif target_units == "ate":
                    est = att * numtreatedunits

                # Return indices in the original dataframe
                self.matched_indices_att = {}
                treated_df_index = treated.index.tolist()
                for i in range(numtreatedunits):
                    self.matched_indices_att[treated_df_index[i]] = control.iloc[indices[i]].index.tolist()
            else:
                grouped = updated_df.groupby(self.exact_match_cols)
                att = 0
                for name, group in grouped:
                    treated = group.loc[group[self._treatment_name[0]] == 1]
                    control = group.loc[group[self._treatment_name[0]] == 0]
                    if treated.shape[0] == 0:
                        continue
                    control_neighbors = NearestNeighbors(
                        n_neighbors=self.num_matches_per_unit,
                        metric=self.distance_metric,
                        algorithm="ball_tree",
                        **self.distance_metric_params,
                    ).fit(control[self._observed_common_causes.columns].values)
                    distances, indices = control_neighbors.kneighbors(
                        treated[self._observed_common_causes.columns].values
                    )
                    self.logger.debug("distances:")
                    self.logger.debug(distances)

                    for i in range(numtreatedunits):
                        treated_outcome = treated.iloc[i][self._outcome_name].item()
                        control_outcome = np.mean(control.iloc[indices[i]][self._outcome_name].values)
                        att += treated_outcome - control_outcome
                        # self.matched_indices_att[treated_df_index[i]] = control.iloc[indices[i]].index.tolist()

                att /= numtreatedunits

                if target_units == "att":
                    est = att
                elif target_units == "ate":
                    est = att * numtreatedunits

        if fit_atc:
            # Now computing ATC
            treated_neighbors = NearestNeighbors(
                n_neighbors=self.num_matches_per_unit,
                metric=self.distance_metric,
                algorithm="ball_tree",
                **self.distance_metric_params,
            ).fit(treated[self._observed_common_causes.columns].values)
            distances, indices = treated_neighbors.kneighbors(control[self._observed_common_causes.columns].values)

            atc = 0
            for i in range(numcontrolunits):
                control_outcome = control.iloc[i][self._outcome_name].item()
                treated_outcome = np.mean(treated.iloc[indices[i]][self._outcome_name].values)
                atc += treated_outcome - control_outcome

            atc /= numcontrolunits

            if target_units == "atc":
                est = atc
            elif target_units == "ate":
                est += atc * numcontrolunits
                est /= numtreatedunits + numcontrolunits

            # Return indices in the original dataframe
            self.matched_indices_atc = {}
            control_df_index = control.index.tolist()
            for i in range(numcontrolunits):
                self.matched_indices_atc[control_df_index[i]] = treated.iloc[indices[i]].index.tolist()

        estimate = CausalEstimate(
            estimate=est,
            control_value=control_value,
            treatment_value=treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
        )

        estimate.add_estimator(self)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
