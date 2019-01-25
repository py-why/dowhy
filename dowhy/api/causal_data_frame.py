import pandas as pd
import logging
from dowhy.do_why import CausalModel


@pd.api.extensions.register_dataframe_accessor("causal")
class CausalAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.use_graph = True

    def _build_model(self, *args, **kwargs):
        if kwargs.get('common_causes'):
            self.use_graph = False
            common_causes = kwargs.get('common_causes')
            del kwargs['common_causes']
        elif kwargs.get('dot_graph'):
            self.use_graph = True
            dot_graph = kwargs.get('dot_graph')
            del kwargs['dot_graph']
        else:
            raise Exception("You must specify a method for determining a backdoor set.")

        if self.use_graph:
            model = CausalModel(data=self._obj,
                                treatment=kwargs["x"],
                                outcome=kwargs["y"],
                                graph=dot_graph)
        else:
            logging.info(self._obj[kwargs["x"]], self._obj[kwargs["y"]], common_causes )
            model = CausalModel(data=self._obj,
                                treatment=kwargs["x"],
                                outcome=kwargs["y"],
                                common_causes=common_causes)
        return model, args, kwargs

    def _do(self, model, *args, **kwargs):
        if kwargs.get('method_name'):
            method_name = kwargs.get('method_name')
            del kwargs['method_name']
        else:
            method_name = "backdoor.linear_regression"
        logging.info("Using {} for estimation.".format(method_name))
        identified_estimand = model.identify_effect()

        df = self._obj.copy()
        f = lambda x: model.do(x, identified_estimand=identified_estimand, method_name=method_name)

        quantity = '$E[{}| do({})]$'.format(kwargs.get('y'), kwargs.get('x'))
        df[quantity] = df[kwargs.get('x')].apply(f)
        kwargs['y'] = quantity
        return df, args, kwargs

    def plot(self, *args, **kwargs):
        model, args, kwargs = self._build_model(*args, **kwargs)
        df, args, kwargs = self._do(model, *args, **kwargs)

        if kwargs.get('kind') == 'bar':
            agg = df.groupby(kwargs.get('x')).mean()
            del kwargs['x']
            agg.plot(*args, **kwargs)
        elif kwargs.get('kind') == 'line' or not kwargs.get('kind'):
            df.plot(*args, **kwargs)
        else:
            raise Exception("Plot type {} not supported for causal plots!".format(kwargs.get('kind')))

    def mean(self, *args, **kwargs):
        model, args, kwargs = self._build_model(*args, **kwargs)
        df, args, kwargs = self._do(model, *args, **kwargs)
        return df.groupby(kwargs.get('x')).mean()
