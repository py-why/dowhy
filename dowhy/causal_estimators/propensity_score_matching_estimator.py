from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator

class PropensityScoreMatchingEstimator(PropensityScoreEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info("INFO: Using Propensity Score Matching Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self, recalculate_propensity_score=False):
        if self._propensity_score_model is None or recalculate_propensity_score is True:
            self._propensity_score_model = linear_model.LogisticRegression(solver="lbfgs")
            self._propensity_score_model.fit(self._observed_common_causes, self._treatment.to_numpy())
            self._data['propensity_score'] = self._propensity_score_model.predict_proba(self._observed_common_causes)[:,1]
            self._data['propensity_score_logit'] = np.log(self._data['propensity_score'] / (1-self._data['propensity_score']))
            # self.radius = 0.2 * np.std(self._data['propensity_score_logit'])

        # this assumes a binary treatment regime
        treated = self._data.loc[self._data[self._treatment_name[0]] == 1]
        control = self._data.loc[self._data[self._treatment_name[0]] == 0]


        # TODO remove neighbors that are more than a given radius apart
        self.radius = 0.2 * np.sqrt((np.var(control['propensity_score_logit'].values)+np.var(treated['propensity_score_logit'].values))/2)


        # estimate ATT on treated by summing over difference between matched neighbors
        # estimate relative risk, (a/(a+b))/(c/(c+d)), and odd ratio (a/b)/(c/d)
        # where a is treatment & positive, b treatment & negative, c control & positve and d control & negative

        control_neighbors = (
            NearestNeighbors(n_neighbors=1, radius=self.radius, algorithm='ball_tree')
            .fit(control['propensity_score_logit'].values.reshape(-1, 1))
        )
        distances, indices = control_neighbors.kneighbors(treated['propensity_score_logit'].values.reshape(-1, 1))
        self.logger.debug("distances:")
        self.logger.debug(distances)
        
        att = 0
        treatment_positive, control_positive = 0, 0
        numtreatedunits = treated.shape[0]
        for i in range(numtreatedunits):
            treated_outcome = treated.iloc[i][self._outcome_name].item()
            control_outcome = control.iloc[indices[i]][self._outcome_name].item()
            if distances[i] <= self.radius:
                if treated_outcome == 1:
                    treatment_positive += 1
                if control_outcome == 1:
                    control_positive += 1
            att += treated_outcome - control_outcome

        att /= numtreatedunits
        relative_risk = (treatment_positive/numtreatedunits) / (control_positive/numtreatedunits)
        odd_ratio = (treatment_positive/(numtreatedunits-treatment_positive)) / (control_positive/(numtreatedunits-control_positive))

        #Now computing ATC
        treated_neighbors = (
            NearestNeighbors(n_neighbors=1, radius=self.radius, algorithm='ball_tree')
            .fit(treated['propensity_score_logit'].values.reshape(-1, 1))
        )
        distances, indices = treated_neighbors.kneighbors(control['propensity_score_logit'].values.reshape(-1, 1))
        atc = 0
        numcontrolunits = control.shape[0]
        for i in range(numcontrolunits):
            control_outcome = control.iloc[i][self._outcome_name].item()
            treated_outcome = treated.iloc[indices[i]][self._outcome_name].item()
            atc += treated_outcome - control_outcome

        atc /= numcontrolunits
        ate = (att*numtreatedunits + atc*numcontrolunits)/(numtreatedunits+numcontrolunits)

        if self._target_units == "att":
            est = att
        elif self._target_units == "relative_risk":
            est = relative_risk
        elif self._target_units == "odd_ratio":
            est = odd_ratio
        elif self._target_units == "atc":
            est = atc
        elif self._target_units == "ate":
            est = ate
        elif self._target_units == "all":
            est = {"att": att, "atc": atc, "ate": ate, "relative_risk": relative_risk, "odd_ratio": odd_ratio}
        else:
            raise ValueError("Target units string value not supported")

        estimate = CausalEstimate(estimate=est,
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
