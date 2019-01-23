import pandas as pd
import logging
from dowhy.do_why import CausalModel


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.use_graph = True

    @property
    def center(self):
        # return the geographic center point of this DataFrame
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

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
