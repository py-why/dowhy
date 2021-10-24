from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator

class PropensityScoreMatchingEstimator(PropensityScoreEstimator):
    """ Estimate effect of treatment by finding matching treated and control units based on propensity score.

    Straightforward application of the back-door criterion.

    Supports additional parameters that can be specified in the estimate_effect() method.

    - 'propensity_score_model': The model used to compute propensity score. Could be any classification model that supports fit() and predict_proba() methods. If None, use LogisticRegression model as the default. Default=None
    - 'recalculate_propensity_score': If true, force the estimator to calculate the propensity score. To use pre-computed propensity score, set this value to false. Default=True
    - 'propensity_score_column': column name that stores the propensity score. Default='propensity_score'

    """
    def __init__(
        self, 
        *args, 
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

        self.logger.info("INFO: Using Propensity Score Matching Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        if self.recalculate_propensity_score is True:
            if self.propensity_score_model is None:
                self.propensity_score_model = linear_model.LogisticRegression()
            self.propensity_score_model.fit(self._observed_common_causes, self._treatment)
            self._data[self.propensity_score_column] = self.propensity_score_model.predict_proba(self._observed_common_causes)[:, 1]
        else:
            # check if the user provides a propensity score column
            if self.propensity_score_column not in self._data.columns:
                raise ValueError(f"Propensity score column {self.propensity_score_column} does not exist. Please specify the column name that has your pre-computed propensity score.")
            else:
                self.logger.info(f"INFO: Using pre-computed propensity score in column {self.propensity_score_column}")


        # this assumes a binary treatment regime
        treated = self._data.loc[self._data[self._treatment_name[0]] == 1]
        control = self._data.loc[self._data[self._treatment_name[0]] == 0]


        # TODO remove neighbors that are more than a given radius apart

        # estimate ATT on treated by summing over difference between matched neighbors
        control_neighbors = (
            NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
            .fit(control[self.propensity_score_column].values.reshape(-1, 1))
        )
        distances, indices = control_neighbors.kneighbors(treated[self.propensity_score_column].values.reshape(-1, 1))
        self.logger.debug("distances:")
        self.logger.debug(distances)

        att = 0
        numtreatedunits = treated.shape[0]
        for i in range(numtreatedunits):
            treated_outcome = treated.iloc[i][self._outcome_name].item()
            control_outcome = control.iloc[indices[i]][self._outcome_name].item()
            att += treated_outcome - control_outcome

        att /= numtreatedunits

        #Now computing ATC
        treated_neighbors = (
            NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
            .fit(treated[self.propensity_score_column].values.reshape(-1, 1))
        )
        distances, indices = treated_neighbors.kneighbors(control[self.propensity_score_column].values.reshape(-1, 1))
        atc = 0
        numcontrolunits = control.shape[0]
        for i in range(numcontrolunits):
            control_outcome = control.iloc[i][self._outcome_name].item()
            treated_outcome = treated.iloc[indices[i]][self._outcome_name].item()
            atc += treated_outcome - control_outcome

        atc /= numcontrolunits

        if self._target_units == "att":
            est = att
        elif self._target_units == "atc":
            est = atc
        elif self._target_units == "ate":
            est = (att*numtreatedunits + atc*numcontrolunits)/(numtreatedunits+numcontrolunits)
        else:
            raise ValueError("Target units string value not supported")

        estimate = CausalEstimate(estimate=est,
                                  control_value=self._control_value,
                                  treatment_value=self._treatment_value,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  propensity_scores=self._data[self.propensity_score_column])
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
