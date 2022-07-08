import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.exceptions import NotFittedError

from dowhy.causal_estimator import CausalEstimator


class PropensityScoreEstimator(CausalEstimator):
    """
    Base class for estimators that estimate effects based on propensity of
    treatment assignment.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """
    def __init__(self, *args, propensity_score_model=None,
                 recalculate_propensity_score=True,
                 propensity_score_column="propensity_score", **kwargs):
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
        # Required to ensure that self.method_params contains all the
        # parameters to create an object of this class
        args_dict = {k: v for k, v in locals().items()
                     if k not in type(self)._STD_INIT_ARGS}
        args_dict.update(kwargs)
        super().__init__(*args, **args_dict)

        # Enable the user to pass params for a custom propensity model
        self.propensity_score_model = propensity_score_model
        self.recalculate_propensity_score = recalculate_propensity_score
        self.propensity_score_column = propensity_score_column

        # Check if the treatment is one-dimensional
        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        # Checking if the treatment is binary
        treatment_values = self._data[self._treatment_name[0]].astype(int).unique()
        if any([v not in [0,1] for v in treatment_values]):
            error_msg = "Propensity score methods are applicable only for binary treatments"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        self.logger.debug("Back-door variables used:" +
                        ",".join(self._target_estimand.get_backdoor_variables()))

        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()

        if self._observed_common_causes_names:
            self._observed_common_causes = self._data[self._observed_common_causes_names]
            # Convert the categorical variables into dummy/indicator variables
            # Basically, this gives a one hot encoding for each category
            # The first category is taken to be the base line.
            self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        else:
            self._observed_common_causes = None
            error_msg = "No common causes/confounders present. Propensity score based methods are not applicable"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _refresh_propensity_score(self):
        if self.recalculate_propensity_score is True:
            if self.propensity_score_model is None:
                self.propensity_score_model = linear_model.LogisticRegression()
            self.propensity_score_model.fit(self._observed_common_causes, self._treatment)
            self._data[self.propensity_score_column] = self.propensity_score_model.predict_proba(self._observed_common_causes)[:, 1]
        else:
            # check if user provides the propensity score column
            if self.propensity_score_column not in self._data.columns:
                if self.propensity_score_model is None:
                    raise ValueError(f"""Propensity score column {self.propensity_score_column} does not exist, nor does a propensity_model. 
                    Please specify the column name that has your pre-computed propensity score, or a model to compute it.""")
                else:
                    try:
                        self._data[self.propensity_score_column] = self.propensity_score_model.predict_proba(
                            self._observed_common_causes)[:, 1]
                    except NotFittedError:
                        raise NotFittedError("Please fit the propensity score model before calling predict_proba")

            else:
                self.logger.info(f"INFO: Using pre-computed propensity score in column {self.propensity_score_column}")

    def construct_symbolic_estimator(self, estimand):
        '''
            A symbolic string that conveys what each estimator does.
            For instance, linear regression is expressed as
            y ~ bx + e
        '''
        raise NotImplementedError

    def _estimate_effect(self):
        '''
            A custom estimator based on the way the propensity score estimates are to be used.

        '''
        raise NotImplementedError

