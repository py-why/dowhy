from sklearn import linear_model

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator


class PropensityScoreStratificationEstimator(CausalEstimator):
    """ Estimate effect of treatment by stratifying the data into bins with
    identical common causes.

    Straightforward application of the back-door criterion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))
        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        self._observed_common_causes = self._data[self._observed_common_causes_names]
        self.logger.info("INFO: Using Propensity Score Stratification Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        self.numStrata = 50
        self.clippingThreshold = 10

    def _estimate_effect(self):
        psmodel = linear_model.LinearRegression()
        psmodel.fit(self._observed_common_causes, self._treatment)
        self._data['ps'] = psmodel.predict(self._observed_common_causes)

        # sort the dataframe by propensity score
        # create a column 'strata' for each element that marks what strata it belongs to
        numrows = self._data[self._outcome_name].shape[0]
        self._data['strata'] = (
            (self._data['ps'].rank(ascending=True) / numrows) * self.numStrata
        ).round(0)

        # for each strata, count how many treated and control units there are
        # throw away strata that have insufficient treatment or control
        # print("before clipping, here is the distribution of treatment and control per strata")
        # print(self._data.groupby(['strata',self._treatment_name])[self._outcome_name].count())

        self._data['dbar'] = 1 - self._data[self._treatment_name]
        self._data['d_y'] = self._data[self._treatment_name] * self._data[self._outcome_name]
        self._data['dbar_y'] = self._data['dbar'] * self._data[self._outcome_name]
        stratified = self._data.groupby('strata')
        clipped = stratified.filter(
            lambda strata: min(strata.loc[strata[self._treatment_name] == 1].shape[0],
                               strata.loc[strata[self._treatment_name] == 0].shape[0]) > self.clippingThreshold
        )
        # print("after clipping at threshold, now we have:" )
        # print(clipped.groupby(['strata',self._treatment_name])[self._outcome_name].count())

        # sum weighted outcomes over all strata  (weight by treated population)
        weightedoutcomes = clipped.groupby('strata').agg({
            self._treatment_name: ['sum'],
            'dbar': ['sum'],
            'd_y': ['sum'],
            'dbar_y': ['sum']
        })
        weightedoutcomes.columns = ["_".join(x) for x in weightedoutcomes.columns.ravel()]
        treatment_sum_name = self._treatment_name + "_sum"

        weightedoutcomes['d_y_mean'] = weightedoutcomes['d_y_sum'] / weightedoutcomes[treatment_sum_name]
        weightedoutcomes['dbar_y_mean'] = weightedoutcomes['dbar_y_sum'] / weightedoutcomes['dbar_sum']
        weightedoutcomes['effect'] = weightedoutcomes['d_y_mean'] - weightedoutcomes['dbar_y_mean']
        totaltreatmentpopulation = weightedoutcomes[treatment_sum_name].sum()

        ate = (weightedoutcomes['effect'] * weightedoutcomes[treatment_sum_name]).sum() / totaltreatmentpopulation

        # TODO - how can we add additional information into the returned estimate?
        #        such as how much clipping was done, or per-strata info for debugging?
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + estimand.outcome_variable + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = [estimand.treatment_variable, ] + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
