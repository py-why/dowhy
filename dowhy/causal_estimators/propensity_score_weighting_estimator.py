import numpy as np
import pandas as pd


from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator


class PropensityScoreWeightingEstimator(PropensityScoreEstimator):
    """ Estimate effect of treatment by weighing the data by
    inverse probability of occurrence.

    Straightforward application of the back-door criterion.

    For a list of standard args and kwargs, see documentation for
    :class:`~dowhy.causal_estimator.CausalEstimator`.

    Supports additional parameters as listed below.

    """
    def __init__(
        self,
        *args,
        min_ps_score=0.05,
        max_ps_score=0.95,
        weighting_scheme='ips_weight',
        propensity_score_model=None,
        recalculate_propensity_score=True,
        propensity_score_column="propensity_score",
        **kwargs):
        """
        :param min_ps_score: Lower bound used to clip the propensity score.
            Default=0.05
        :param max_ps_score: Upper bound used to clip the propensity score.
            Default=0.95
        :param weighting_scheme: Weighting method to use. Can be inverse
            propensity score ("ips_weight", default), stabilized IPS score
            ("ips_stabilized_weight"), or normalized IPS score
            ("ips_normalized_weight").
        :param propensity_score_model: The model used to compute propensity
            score. Can be any classification model that supports fit() and
            predict_proba() methods. If None, use LogisticRegression model as
            the default. Default=None
        :param recalculate_propensity_score: If true, force the estimator to
            estimate the propensity score. To use pre-computed propensity
            scores, set this value to false. Default=True
        :param propensity_score_column: Column name that stores the
            propensity score. Default='propensity_score'
        """
        # Required to ensure that self.method_params contains all the information
        # to create an object of this class
        args_dict = kwargs
        args_dict.update({
            'min_ps_score': min_ps_score,
            'max_ps_score': max_ps_score,
            'weighting_scheme': weighting_scheme
            })
        super().__init__(
            *args,
            propensity_score_model=propensity_score_model,
            recalculate_propensity_score=recalculate_propensity_score,
            propensity_score_column=propensity_score_column,
            **args_dict)

        self.logger.info("INFO: Using Propensity Score Weighting Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(
            self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        # Setting method specific parameters
        self.weighting_scheme = weighting_scheme
        self.min_ps_score = min_ps_score
        self.max_ps_score = max_ps_score

    def _estimate_effect(self):
        self._refresh_propensity_score()

        # trim propensity score weights
        self._data[self.propensity_score_column] = np.minimum(self.max_ps_score, self._data[self.propensity_score_column])
        self._data[self.propensity_score_column] = np.maximum(self.min_ps_score, self._data[self.propensity_score_column])

        # ips ==> (isTreated(y)/ps(y)) + ((1-isTreated(y))/(1-ps(y)))
        # nips ==> ips / (sum of ips over all units)
        # icps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all control units)
        # itps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all treatment units)
        ipst_sum = sum(self._data[self._treatment_name[0]] / self._data[self.propensity_score_column])
        ipsc_sum = sum((1 - self._data[self._treatment_name[0]]) / (1-self._data[self.propensity_score_column]))
        num_units = len(self._data[self._treatment_name[0]])
        num_treatment_units = sum(self._data[self._treatment_name[0]])
        num_control_units = num_units - num_treatment_units

        # Vanilla IPS estimator
        self._data['ips_weight'] = (
            self._data[self._treatment_name[0]] / self._data[self.propensity_score_column] +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data[self.propensity_score_column])
        )
        self._data['tips_weight'] = (
            self._data[self._treatment_name[0]] +
            (1 - self._data[self._treatment_name[0]]) * self._data[self.propensity_score_column] / (1 - self._data[self.propensity_score_column])
        )
        self._data['cips_weight'] = (
            self._data[self._treatment_name[0]] * (1 - self._data[self.propensity_score_column]) / self._data[self.propensity_score_column] + (1 - self._data[self._treatment_name[0]])
        )

        # The Hajek estimator (or the self-normalized estimator)
        self._data['ips_normalized_weight'] = (
            self._data[self._treatment_name[0]] / self._data[self.propensity_score_column] / ipst_sum +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data[self.propensity_score_column]) / ipsc_sum
        )
        ipst_for_att_sum = sum(self._data[self._treatment_name[0]])
        ipsc_for_att_sum = sum((1-self._data[self._treatment_name[0]])/(1 - self._data[self.propensity_score_column])*self._data[self.propensity_score_column])
        self._data['tips_normalized_weight'] = (
            self._data[self._treatment_name[0]] / ipst_for_att_sum +
            (1 - self._data[self._treatment_name[0]]) * self._data[self.propensity_score_column] / (1 - self._data[self.propensity_score_column]) / ipsc_for_att_sum
        )
        ipst_for_atc_sum = sum(self._data[self._treatment_name[0]] / self._data[self.propensity_score_column] * (1-self._data[self.propensity_score_column]))
        ipsc_for_atc_sum = sum((1 - self._data[self._treatment_name[0]]))
        self._data['cips_normalized_weight'] = (
            self._data[self._treatment_name[0]] * (1 - self._data[self.propensity_score_column]) / self._data[self.propensity_score_column] / ipst_for_atc_sum +
            (1 - self._data[self._treatment_name[0]])/ipsc_for_atc_sum
        )

        # Stabilized weights (from Robins, Hernan, Brumback (2000))
        # Paper: Marginal Structural Models and Causal Inference in Epidemiology
        p_treatment = sum(self._data[self._treatment_name[0]])/num_units
        self._data['ips_stabilized_weight'] = (
            self._data[self._treatment_name[0]] / self._data[self.propensity_score_column] * p_treatment +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data[self.propensity_score_column]) * (1 - p_treatment)
        )
        self._data['tips_stabilized_weight'] = (
            self._data[self._treatment_name[0]] * p_treatment +
            (1 - self._data[self._treatment_name[0]]) * self._data[self.propensity_score_column] / (1 - self._data[self.propensity_score_column]) * (1 - p_treatment)
        )
        self._data['cips_stabilized_weight'] = (
            self._data[self._treatment_name[0]] * (1 - self._data[self.propensity_score_column]) / self._data[self.propensity_score_column] * p_treatment +
            (1 - self._data[self._treatment_name[0]]) * (1-p_treatment)
        )

        if self._target_units == "ate":
            weighting_scheme_name = self.weighting_scheme
        elif self._target_units == "att":
            weighting_scheme_name = "t" + self.weighting_scheme
        elif self._target_units == "atc":
            weighting_scheme_name = "c" + self.weighting_scheme
        else:
            raise ValueError("Target units string value not supported")

        # Calculating the effect
        self._data['d_y'] = (
            self._data[weighting_scheme_name] *
            self._data[self._treatment_name[0]] *
            self._data[self._outcome_name]
        )
        self._data['dbar_y'] = (
            self._data[weighting_scheme_name] *
            (1 - self._data[self._treatment_name[0]]) *
            self._data[self._outcome_name]
        )
        sum_dy_weights = np.sum(
            self._data[self._treatment_name[0]] * self._data[weighting_scheme_name])
        sum_dbary_weights = np.sum(
            (1 - self._data[self._treatment_name[0]]) * self._data[weighting_scheme_name])
        # Subtracting the weighted means
        est = self._data['d_y'].sum() / sum_dy_weights - self._data['dbar_y'].sum() / sum_dbary_weights

        # TODO - how can we add additional information into the returned estimate?
        estimate = CausalEstimate(estimate=est,
                                  control_value=self._control_value,
                                  treatment_value=self._treatment_value,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  propensity_scores=self._data[self.propensity_score_column])
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
