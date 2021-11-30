from sklearn import linear_model
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator


class PropensityScoreStratificationEstimator(PropensityScoreEstimator):
    """ Estimate effect of treatment by stratifying the data into bins with
    identical common causes.

    Straightforward application of the back-door criterion.

    Supports additional parameters that can be specified in the estimate_effect() method.

    - 'num_strata': Number of bins by which data will be stratified. Default=50
    - 'clipping_threshold': Mininum number of treated or control units per strata. Default=10
    - 'propensity_score_model': The model used to compute propensity score. Could be any classification model that supports fit() and predict_proba() methods. If None, use LogisticRegression model as the default. Default=None
    - 'recalculate_propensity_score': If true, force the estimator to calculate the propensity score. To use pre-computed propensity score, set this value to false. Default=True
    - 'propensity_score_column': column name that stores the propensity score. Default='propensity_score'

    """

    def __init__(
        self,
        *args,
        num_strata="auto",
        clipping_threshold=10,
        propensity_score_model=None,
        recalculate_propensity_score=True,
        propensity_score_column="propensity_score",
        **kwargs):
        super().__init__(
            *args,
            propensity_score_model=propensity_score_model,
            recalculate_propensity_score=recalculate_propensity_score,
            propensity_score_column=propensity_score_column,
            **kwargs)

        self.logger.info("Using Propensity Score Stratification Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        if not hasattr(self, 'num_strata'):
            self.num_strata = num_strata
        if not hasattr(self, 'clipping_threshold'):
            self.clipping_threshold = clipping_threshold


    def _estimate_effect(self):
        if self.recalculate_propensity_score is True:
            if self.propensity_score_model is None:
                self.propensity_score_model = linear_model.LogisticRegression()
            self.propensity_score_model.fit(self._observed_common_causes, self._treatment)
            self._data[self.propensity_score_column] = self.propensity_score_model.predict_proba(self._observed_common_causes)[:, 1]
        else:
            # check if the user provides the propensity score column
            if self.propensity_score_column not in self._data.columns:
                raise ValueError(f"Propensity score column {self.propensity_score_column} does not exist. Please specify the column name that has your pre-computed propensity score.")
            else:
                self.logger.info(f"Using pre-computed propensity score incolumn {self.propensity_score_column}")
        clipped = None
        # Infer the right strata based on clipping threshold
        if self.num_strata == "auto":
            # 0.5 because there are two values for the treatment
            clipping_t = self.clipping_threshold
            num_strata = 0.5 * self._data.shape[0] / clipping_t
            # To be conservative and allow most strata to be included in the
            # analysis
            strata_found = False
            while not strata_found:
                self.logger.info("'num_strata' selected as {}".format(num_strata))
                try:
                    clipped = self._get_strata(num_strata, self.clipping_threshold)
                    num_ret_strata = clipped.groupby(['strata']).count().reset_index()
                    # At least 90% of the strata should be included in analysis
                    if num_ret_strata.shape[0] >= 0.5 * num_strata:
                        strata_found = True
                    else:
                        num_strata = int(num_strata / 2)
                        self.logger.info(f"Less than half the strata have at least {self.clipping_threshold} data points. Selecting fewer number of strata.")
                        if num_strata < 2:
                            raise ValueError("Not enough data to generate at least two strata. This error may be due to a high value of 'clipping_threshold'.")
                except ValueError:
                    self.logger.info("No strata found with at least {} data points. Selecting fewer number of strata".format(self.clipping_threshold))
                    num_strata = int(num_strata / 2)
                    if num_strata < 2:
                        raise ValueError("Not enough data to generate at least two strata. This error may be due to a high value of 'clipping_threshold'.")
        else:
            clipped = self._get_strata(self.num_strata, self.clipping_threshold)

        # sum weighted outcomes over all strata  (weight by treated population)
        weighted_outcomes = clipped.groupby('strata').agg({
            self._treatment_name[0]: ['sum'],
            'dbar': ['sum'],
            'd_y': ['sum'],
            'dbar_y': ['sum']
        })
        weighted_outcomes.columns = ["_".join(x) for x in weighted_outcomes.columns.to_numpy().ravel()]
        treatment_sum_name = self._treatment_name[0] + "_sum"
        control_sum_name = "dbar_sum"

        weighted_outcomes['d_y_mean'] = weighted_outcomes['d_y_sum'] / weighted_outcomes[treatment_sum_name]
        weighted_outcomes['dbar_y_mean'] = weighted_outcomes['dbar_y_sum'] / weighted_outcomes['dbar_sum']
        weighted_outcomes['effect'] = weighted_outcomes['d_y_mean'] - weighted_outcomes['dbar_y_mean']
        total_treatment_population = weighted_outcomes[treatment_sum_name].sum()
        total_control_population = weighted_outcomes[control_sum_name].sum()
        total_population = total_treatment_population + total_control_population
        self.logger.debug("Total number of data points is {0}, including {1} from treatment and {2} from control.". format(total_population, total_treatment_population, total_control_population))

        if self._target_units=="att":
            est = (weighted_outcomes['effect'] * weighted_outcomes[treatment_sum_name]).sum() / total_treatment_population
        elif self._target_units=="atc":
            est = (weighted_outcomes['effect'] * weighted_outcomes[control_sum_name]).sum() / total_control_population
        elif self._target_units == "ate":
            est = (weighted_outcomes['effect'] * (weighted_outcomes[control_sum_name]+weighted_outcomes[treatment_sum_name])).sum() / total_population
        else:
            raise ValueError("Target units string value not supported")

        # TODO - how can we add additional information into the returned estimate?
        #        such as how much clipping was done, or per-strata info for debugging?
        estimate = CausalEstimate(estimate=est,
                                  control_value=self._control_value,
                                  treatment_value=self._treatment_value,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  propensity_scores = self._data[self.propensity_score_column])
        return estimate

    def _get_strata(self, num_strata, clipping_threshold):
        # sort the dataframe by propensity score
        # create a column 'strata' for each element that marks what strata it belongs to
        num_rows = self._data[self._outcome_name].shape[0]
        self._data['strata'] = (
            (self._data[self.propensity_score_column].rank(ascending=True) / num_rows) * num_strata
        ).round(0)
        # for each strata, count how many treated and control units there are
        # throw away strata that have insufficient treatment or control

        self._data['dbar'] = 1 - self._data[self._treatment_name[0]] # 1-Treatment
        self._data['d_y'] = self._data[self._treatment_name[0]] * self._data[self._outcome_name]
        self._data['dbar_y'] = self._data['dbar'] * self._data[self._outcome_name]
        stratified = self._data.groupby('strata')
        clipped = stratified.filter(
            lambda strata: min(strata.loc[strata[self._treatment_name[0]] == 1].shape[0],
                               strata.loc[strata[self._treatment_name[0]] == 0].shape[0]) > clipping_threshold
        )
        self.logger.debug("After using clipping_threshold={0}, here are the number of data points in each strata:\n {1}".format(clipping_threshold, clipped.groupby(['strata',self._treatment_name[0]])[self._outcome_name].count()))
        if clipped.empty:
            raise ValueError("Method requires strata with number of data points per treatment > clipping_threshold (={0}). No such strata exists. Consider decreasing 'num_strata' or 'clipping_threshold' parameters.".format(clipping_threshold))
        return clipped

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
