import numpy as np
import pandas as pd

from dowhy.causal_estimator import CausalEstimator

class PropensityScoreEstimator(CausalEstimator):
    """ 
    Base class for estimators that estimate effects based on propensity of treatment assignment.
    Supports additional parameters that can be specified in the estimate_effect() method.
    - 'propensity_score_model': The model used to compute propensity score. Could be any classification model that supports fit() and predict_proba() methods. If None, use LogisticRegression model as the default. Default=None
    - 'recalculate_propensity_score': If true, force the estimator to calculate the propensity score. To use pre-computed propensity score, set this value to false. Default=True
    - 'propensity_score_column': column name that stores the propensity score. Default='propensity_score'
    """
    def __init__(self, *args, propensity_score_model=None, recalculate_propensity_score=True, propensity_score_column="propensity_score", **kwargs):
        super().__init__(*args, **kwargs)

        # Enable the user to pass params for a custom propensity model
        if not hasattr(self, "propensity_score_model"):
            self.propensity_score_model = propensity_score_model
        if not hasattr(self, "recalculate_propensity_score"):
            self.recalculate_propensity_score = recalculate_propensity_score
        if not hasattr(self, "propensity_score_column"):
            self.propensity_score_column = propensity_score_column

        # Check if the treatment is one-dimensional
        if len(self._treatment_name) > 1:
            error_msg = str(self.__class__) + "cannot handle more than one treatment variable"
            raise Exception(error_msg)
        # Checking if the treatment is binary
        if not pd.api.types.is_bool_dtype(self._data[self._treatment_name[0]]):
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

