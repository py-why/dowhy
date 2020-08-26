import numpy as np
import pandas as pd
from sklearn import linear_model

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator

class PropensityScoreWeightingEstimator(PropensityScoreEstimator):
    """ Estimate effect of treatment by weighing the data by
    inverse probability of occurrence.

    Straightforward application of the back-door criterion.

    Supports additional parameters that can be specified in the estimate_effect() method.

    - 'weighting_scheme': This is the name of weighting method to use. Can be inverse propensity score ("ips_weight", default), stabilized IPS score ("ips_stabilized_weight"), or normalized IPS score ("ips_normalized_weight")

    """

    def __init__(self, *args, min_ps_score=0.05, max_ps_score=0.95, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info("INFO: Using Propensity Score Weighting Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)
        if not hasattr(self, "weighting_scheme"):
            self.weighting_scheme = 'ips_weight'  # 'ips_weight', 'ips_normalized_weight', 'ips_stabilized_weight'
        self.min_ps_score = min_ps_score
        self.max_ps_score = max_ps_score

    def _estimate_effect(self, recalculate_propensity_score=False):
        if self._propensity_score_model is None or recalculate_propensity_score is True:
            self._propensity_score_model = linear_model.LogisticRegression()
            self._propensity_score_model.fit(self._observed_common_causes, self._treatment)
            self._data['ps'] = self._propensity_score_model.predict_proba(self._observed_common_causes)[:,1]
        
        # trim propensity score weights
        self._data['ps'] = np.minimum(self.max_ps_score, self._data['ps'])
        self._data['ps'] = np.maximum(self.min_ps_score, self._data['ps'])

        # ips ==> (isTreated(y)/ps(y)) + ((1-isTreated(y))/(1-ps(y)))
        # nips ==> ips / (sum of ips over all units)
        # icps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all control units)
        # itps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all treatment units)
        ipst_sum = sum(self._data[self._treatment_name[0]] / self._data['ps'])
        ipsc_sum = sum((1 - self._data[self._treatment_name[0]]) / (1-self._data['ps']))
        num_units = len(self._data[self._treatment_name[0]])
        num_treatment_units = sum(self._data[self._treatment_name[0]])
        num_control_units = num_units - num_treatment_units
        # Vanilla IPS estimator

        self._data['ips_weight'] = (1/num_units) * (
            self._data[self._treatment_name[0]] / self._data['ps'] +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data['ps'])
        )
        self._data['tips_weight'] = (1/num_treatment_units) * (
            self._data[self._treatment_name[0]] +
            (1 - self._data[self._treatment_name[0]]) * self._data['ps']/ (1 - self._data['ps'])
        )
        self._data['cips_weight'] = (1/num_control_units) * (
            self._data[self._treatment_name[0]] * (1 - self._data['ps'])/ self._data['ps'] +
            (1 - self._data[self._treatment_name[0]])
        )

        # Also known as the Hajek estimator
        self._data['ips_normalized_weight'] = (
            self._data[self._treatment_name[0]] / self._data['ps'] / ipst_sum +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data['ps']) / ipsc_sum
        )
        ipst_for_att_sum = sum(self._data[self._treatment_name[0]])
        ipsc_for_att_sum = sum((1-self._data[self._treatment_name[0]])/(1 - self._data['ps'])*self._data['ps'] )
        self._data['tips_normalized_weight'] = (
            self._data[self._treatment_name[0]]/ ipst_for_att_sum  +
            (1 - self._data[self._treatment_name[0]]) * self._data['ps'] / (1 - self._data['ps']) / ipsc_for_att_sum
        )
        ipst_for_atc_sum = sum(self._data[self._treatment_name[0]] / self._data['ps'] * (1-self._data['ps']))
        ipsc_for_atc_sum = sum((1 - self._data[self._treatment_name[0]]))
        self._data['cips_normalized_weight'] = (
            self._data[self._treatment_name[0]] * (1 - self._data['ps']) / self._data['ps'] / ipst_for_atc_sum +
            (1 - self._data[self._treatment_name[0]])/ipsc_for_atc_sum
        )

        # Stabilized weights
        p_treatment = sum(self._data[self._treatment_name[0]])/num_units
        self._data['ips_stabilized_weight'] = (1/num_units) * (
            self._data[self._treatment_name[0]] / self._data['ps'] * p_treatment +
            (1 - self._data[self._treatment_name[0]]) / (1 - self._data['ps']) * (1- p_treatment)
        )
        self._data['tips_stabilized_weight'] = (1/num_treatment_units) * (
            self._data[self._treatment_name[0]] * p_treatment  +
            (1 - self._data[self._treatment_name[0]]) * self._data['ps'] / (1 - self._data['ps']) * (1- p_treatment)
        )
        self._data['cips_stabilized_weight'] = (1/num_control_units) * (
            self._data[self._treatment_name[0]] * (1 - self._data['ps']) / self._data['ps'] * p_treatment +
            (1 - self._data[self._treatment_name[0]])* (1-p_treatment)
        )

        # Simple normalized estimator (commented out for now)
        #ips_sum = self._data['ips_weight'].sum()
        #self._data['nips_weight'] = self._data['ips_weight'] / ips_sum

        #self._data['ips2'] = self._data['ps'] / (1 - self._data['ps'])
        #treated_ips_sum = (self._data['ips2'] * self._data[self._treatment_name[0]]).sum()
        #control_ips_sum = (self._data['ips2'] * (1 - self._data[self._treatment_name[0]])).sum()
        #self._data['itps_weight'] = self._data['ips2'] / treated_ips_sum
        #self._data['icps_weight'] = self._data['ips2'] / control_ips_sum

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
        est = self._data['d_y'].sum() - self._data['dbar_y'].sum()


        # TODO - how can we add additional information into the returned estimate?
        estimate = CausalEstimate(estimate=est,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator,
                                  propensity_scores = self._data["ps"])
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        return expr
