import pandas as pd
import logging
from dowhy.do_why import CausalModel
import dowhy.do_samplers as do_samplers


class CausalDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_model = None

    def attach_causal_model(self, model):
        self._causal_model = model


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.use_graph = True

    def do(self, x, method=None, num_cores=1, variable_types={}, outcome=None, params=None, graph=None,
           common_causes=None, instruments=None, estimand_type='ate', proceed_when_unidentifiable=False):
        if not method:
            raise Exception("You must specify a do sampling method.")
        if not self._obj._causal_model:
            self._obj._causal_model = CausalModel(self._obj, [xi for xi in x.keys()][0], outcome, graph=graph,
                                             common_causes=common_causes, instruments=instruments,
                                             estimand_type=estimand_type,
                                             proceed_when_unidentifiable=proceed_when_unidentifiable)
        identified_estimand = self._obj._causal_model.identify_effect()
        do_sampler_class = do_samplers.get_class_object(method + "_sampler")
        do_sampler = do_sampler_class(self.pandas_obj,
                                      identified_estimand,
                                      self._obj._causal_model.treatment,
                                      self._obj._causal_model.outcome,
                                      params=params,
                                      variable_types=variable_types,
                                      num_cores=num_cores)


    def plot(self, *args, **kwargs):
        if kwargs.get('method_name'):
            method_name = kwargs.get('method_name')
        else:
            method_name = "backdoor.propensity_score_matching"
        logging.info("Using {} for estimation.".format(method_name))

        if kwargs.get('common_causes'):
            self.use_graph = False
        elif kwargs.get('dot_graph'):
            self.use_graph = True
        else:
            raise Exception("You must specify a method for determining a backdoor set.")

        if self.use_graph:
            model = CausalModel(data=self._obj,
                                treatment=self._obj[kwargs["treatment_name"]],
                                outcome=self._obj[kwargs["outcome_name"]],
                                graph=args["dot_graph"])
        else:
            model = CausalModel(data=self._obj,
                                treatment=self._obj[kwargs["treatment_name"]],
                                outcome=self._obj[kwargs["outcome_name"]],
                                common_causes=args["common_causes"])
        if kwargs['kind'] == 'bar':
            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(identified_estimand,
                                             method_name=method_name)
        elif kwargs['kind'] == 'line' or not kwargs['kind'].get():
            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(identified_estimand,
                                             method_name=method_name)
        else:
            raise Exception("Plot type {} not supported for causal plots!".format(kwargs.get('kind')))
        self._obj.plot(*args, **kwargs)


    def mean(self, *args, **kwargs):
        pass
