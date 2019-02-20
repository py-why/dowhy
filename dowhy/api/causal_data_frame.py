import pandas as pd
import logging
from dowhy.do_why import CausalModel
import dowhy.do_samplers as do_samplers


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._causal_model = None
        self._sampler = None
        self._identified_estimand = None
        self._method = None

    def reset(self):
        self._causal_model = None
        self._identified_estimand = None
        self._sampler = None
        self._method = None

    def do(self, x, method='weighting', num_cores=1, variable_types={}, outcome=None, params=None, dot_graph=None,
           common_causes=None, instruments=None, estimand_type='ate', proceed_when_unidentifiable=False,
           keep_original_treatment=False, stateful=False):
        if not stateful or method != self._method:
            self.reset()
        if not self._causal_model:
            self._causal_model = CausalModel(self._obj,
                                                  [xi for xi in x.keys()][0],
                                                  outcome,
                                                  graph=dot_graph,
                                                  common_causes=common_causes,
                                                  instruments=instruments,
                                                  estimand_type=estimand_type,
                                                  proceed_when_unidentifiable=proceed_when_unidentifiable)
        self._identified_estimand = self._causal_model.identify_effect()
        if not self._sampler:
            self._method = method
            do_sampler_class = do_samplers.get_class_object(method + "_sampler")
            self._sampler = do_sampler_class(self._obj,
                                                  self._identified_estimand,
                                                  self._causal_model._treatment,
                                                  self._causal_model._outcome,
                                                  params=params,
                                                  variable_types=variable_types,
                                                  num_cores=num_cores,
                                                  causal_model=self._causal_model,
                                                  keep_original_treatment=keep_original_treatment)
        result = self._sampler.do_sample(x)
        if not stateful:
            self.reset()
        return result
