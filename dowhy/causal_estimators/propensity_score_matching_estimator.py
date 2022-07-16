from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator

class PropensityScoreMatchingEstimator(PropensityScoreEstimator):
    """ Estimate effect of treatment by finding matching treated and control
    units based on propensity score.

    Straightforward application of the back-door criterion.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """
    def __init__(
        self,
        *args,
        propensity_score_model=None,
        recalculate_propensity_score=True,
        propensity_score_column="propensity_score",
        **kwargs):
        """
        :param propensity_score_model: Model used to compute propensity score.
            Can be any classification model that supports fit() and
            predict_proba() methods. If None, LogisticRegression is used.
        :param recalculate_propensity_score: Whether the propensity score
            should be estimated. To use pre-computed propensity scores,
            set this value to False. Default=True.
        :param propensity_score_column: Column name that stores the
            propensity score. Default='propensity_score'

        """
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
        self._refresh_propensity_score()

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
