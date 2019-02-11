import pandas as pd
import logging
from dowhy.do_why import CausalModel
import dowhy.do_samplers as do_samplers


class CausalDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_model = None
        self._sampler = None
        self._identified_estimand = None

    def attach_causal_model(self, model):
        self._causal_model = model

    def reset(self):
        self._causal_model = None
        self._identified_estimand = None
        self._sampler = None


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def do(self, x, method=None, num_cores=1, variable_types={}, outcome=None, params=None, dot_graph=None,
           common_causes=None, instruments=None, estimand_type='ate', proceed_when_unidentifiable=False,
           keep_original_treatment=False, use_previous_sampler=False):
        if not method:
            raise Exception("You must specify a do sampling method.")
        if not self._obj._causal_model or not use_previous_sampler:
            self._obj._causal_model = CausalModel(self._obj,
                                                  [xi for xi in x.keys()][0],
                                                  outcome,
                                                  graph=dot_graph,
                                                  common_causes=common_causes,
                                                  instruments=instruments,
                                                  estimand_type=estimand_type,
                                                  proceed_when_unidentifiable=proceed_when_unidentifiable)
        self._obj._identified_estimand = self._obj._causal_model.identify_effect()
        do_sampler_class = do_samplers.get_class_object(method + "_sampler")
        if not self._obj._sampler or not use_previous_sampler:
            self._obj._sampler = do_sampler_class(self._obj,
                                                  self._obj._identified_estimand,
                                                  self._obj._causal_model._treatment,
                                                  self._obj._causal_model._outcome,
                                                  params=params,
                                                  variable_types=variable_types,
                                                  num_cores=num_cores,
                                                  causal_model=self._obj._causal_model,
                                                  keep_original_treatment=keep_original_treatment)
        return self._obj._sampler.do_sample(x)


