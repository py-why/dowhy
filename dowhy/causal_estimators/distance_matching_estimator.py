from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimate, CausalEstimator

class DistanceMatchingEstimator(CausalEstimator):
    """ Simple matching estimator for binary treatments based on a distance metric.
    """

    Valid_Dist_Metric_Params = ['p', 'V', 'VI', 'w']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if the treatment is one-dimensional
        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        # Checking if the treatment is binary
        if not pd.api.types.is_bool_dtype(self._data[self._treatment_name[0]]):
            error_msg = "Distance Matching method is applicable only for binary treatments"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Setting the number of matches per data point
        if getattr(self, 'num_matches_per_unit', None) is None:
            self.num_matches_per_unit = 1
        # Default distance metric if not provided by the user
        if getattr(self, 'distance_metric', None) is None:
            self.distance_metric = 'minkowski' # corresponds to euclidean metric with p=2

        if getattr(self, 'exact_match_cols', None) is None:
            self.exact_match_cols = None

        self.logger.debug("Back-door variables used:" +
                        ",".join(self._target_estimand.get_backdoor_variables()))

        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()
        if self._observed_common_causes_names:
            if self.exact_match_cols is not None:
                self._observed_common_causes_names = [v for v in self._observed_common_causes_names if v not in self.exact_match_cols]
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


        # Dictionary of any user-provided params for the distance metric
        # that will be passed to sklearn nearestneighbors
        self.distance_metric_params = {}
        for param_name in self.Valid_Dist_Metric_Params:
            param_val = getattr(self, param_name, None)
            if param_val is not None:
                self.distance_metric_params[param_name] = param_val


        self.logger.info("INFO: Using Distance Matching Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        self.matched_indices_att = None
        self.matched_indices_atc = None

    def _estimate_effect(self):
        # this assumes a binary treatment regime
        updated_df = pd.concat([self._observed_common_causes,
            self._data[[self._outcome_name, self._treatment_name[0]]]], axis=1)
        if self.exact_match_cols is not None:
            updated_df = pd.concat([updated_df, self._data[self.exact_match_cols]], axis=1)
        treated = updated_df.loc[self._data[self._treatment_name[0]] == 1]
        control = updated_df.loc[self._data[self._treatment_name[0]] == 0]
        numtreatedunits = treated.shape[0]
        numcontrolunits = control.shape[0]

        fit_att, fit_atc = False, False
        est = None
        # TODO remove neighbors that are more than a given radius apart
        if self._target_units == "att":
            fit_att = True
        elif self._target_units == "atc":
            fit_atc = True
        elif self._target_units == "ate":
            fit_att = True
            fit_atc = True
        else:
            raise ValueError("Target units string value not supported")

        if fit_att:
            # estimate ATT on treated by summing over difference between matched neighbors
            if self.exact_match_cols is None:
                control_neighbors = (
                    NearestNeighbors(n_neighbors=self.num_matches_per_unit,
                        metric=self.distance_metric,
                        algorithm='ball_tree',
                        **self.distance_metric_params)
                    .fit(control[self._observed_common_causes.columns].values)
                )
                distances, indices = control_neighbors.kneighbors(
                        treated[self._observed_common_causes.columns].values)
                self.logger.debug("distances:")
                self.logger.debug(distances)

                att = 0

                for i in range(numtreatedunits):
                    treated_outcome = treated.iloc[i][self._outcome_name].item()
                    control_outcome = np.mean(control.iloc[indices[i]][self._outcome_name].values)
                    att += treated_outcome - control_outcome

                att /= numtreatedunits
                if self._target_units == "att":
                    est = att
                elif self._target_units == "ate":
                    est = att*numtreatedunits

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
                    control_neighbors = (
                        NearestNeighbors(n_neighbors=self.num_matches_per_unit,
                            metric=self.distance_metric,
                            algorithm='ball_tree',
                            **self.distance_metric_params)
                        .fit(control[self._observed_common_causes.columns].values)
                    )
                    distances, indices = control_neighbors.kneighbors(
                            treated[self._observed_common_causes.columns].values)
                    self.logger.debug("distances:")
                    self.logger.debug(distances)

                    for i in range(numtreatedunits):
                        treated_outcome = treated.iloc[i][self._outcome_name].item()
                        control_outcome = np.mean(control.iloc[indices[i]][self._outcome_name].values)
                        att += treated_outcome - control_outcome
                        #self.matched_indices_att[treated_df_index[i]] = control.iloc[indices[i]].index.tolist()

                att /= numtreatedunits

                if self._target_units == "att":
                    est = att
                elif self._target_units == "ate":
                    est = att*numtreatedunits

        if fit_atc:
            #Now computing ATC
            treated_neighbors = (
                NearestNeighbors(n_neighbors=self.num_matches_per_unit,
                    metric=self.distance_metric,
                    algorithm='ball_tree',
                    **self.distance_metric_params)
                .fit(treated[self._observed_common_causes.columns].values)
            )
            distances, indices = treated_neighbors.kneighbors(
                    control[self._observed_common_causes.columns].values)

            atc = 0
            for i in range(numcontrolunits):
                control_outcome = control.iloc[i][self._outcome_name].item()
                treated_outcome = np.mean(treated.iloc[indices[i]][self._outcome_name].values)
                atc += treated_outcome - control_outcome

            atc /= numcontrolunits

            if self._target_units == "atc":
                est = atc
            elif self._target_units == "ate":
                est += atc*numcontrolunits
                est /= (numtreatedunits+numcontrolunits)

            # Return indices in the original dataframe
            self.matched_indices_atc = {}
            control_df_index = control.index.tolist()
            for i in range(numcontrolunits):
                self.matched_indices_atc[control_df_index[i]] = treated.iloc[indices[i]].index.tolist()

        estimate = CausalEstimate(estimate=est,
                                  control_value=self._control_value,
                                  treatment_value=self._treatment_value,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
