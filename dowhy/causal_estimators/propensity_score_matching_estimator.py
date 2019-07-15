from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator


class PropensityScoreMatchingEstimator(CausalEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))
        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        if self._observed_common_causes_names:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes= None
            error_msg ="No common causes/confounders present. Propensity score based methods are not applicable"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        self.logger.info("INFO: Using Propensity Score Matching Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        propensity_score_model = linear_model.LogisticRegression(solver="lbfgs")
        propensity_score_model.fit(self._observed_common_causes, self._treatment)
        self._data['propensity_score'] = propensity_score_model.predict_proba(self._observed_common_causes)[:,1]

        # this assumes a binary treatment regime
        treated = self._data.loc[self._data[self._treatment_name] == 1]
        control = self._data.loc[self._data[self._treatment_name] == 0]

        control_neighbors = (
            NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
            .fit(control['propensity_score'].values.reshape(-1, 1))
        )
        distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))

        # TODO remove neighbors that are more than a given radius apart

        # estimate ATE on treated by summing over difference between matched neighbors
        ate = 0
        numtreatedunits = treated.shape[0]
        for i in range(numtreatedunits):
            treated_outcome = treated.iloc[i][self._outcome_name].item()
            control_outcome = control.iloc[indices[i]][self._outcome_name].item()
            ate += treated_outcome - control_outcome

        ate /= numtreatedunits

        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  propensity_scores=self._data["propensity_score"])
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
