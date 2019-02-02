import logging
import numpy as np
import pandas as pd


class DoSampler:
    """Base class for a sampler from the interventional distribution.

    """

    def __init__(self, data, identified_estimand, treatments, outcomes, params=None, variable_types=None,
                 num_cores=1):
        """Initializes a do sampler with data and names of relevant variables.

        More description.

        :param data: data frame containing the data
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param treatments: names of the treatment variables
        :param outcomes: names of the outcome variables
        :param params: (optional) additional method parameters
        :returns: an instance of the estimator class.

        """
        self._data = data
        self._target_estimand = identified_estimand
        self._treatment_names = treatments
        self._outcome_names = outcomes
        self._estimate = None
        self._variable_types = variable_types
        self.num_cores = num_cores
        self.point_sampler = True
        self.sampler = None

        if params is not None:
            for key, value in params.items():
                setattr(self, key, value)

        self._df = None

        if not self._variable_types and kwargs.get('variables_types'):
            self._variable_types = kwargs.get('variable_types')
        elif not self._variable_types:
            self._infer_variable_types()
        self.dep_type = [self._variable_types[var] for var in self._outcome_names]
        self.indep_type = [self._variable_types[var] for var in
                           self._treatment_names + self._target_estimand.backdoor_variables]
        self.density_types = [self._variable_types[var] for var in self._target_estimand.backdoor_variables]

        self.outcome_lower_support = self._data[self._outcome_names].min().values
        self.outcome_upper_support = self._data[self._outcome_names].max().values

        self.logger = logging.getLogger(__name__)

    def _sample_point(self, x_z):
        raise NotImplementedError

    def reset(self):
        self._df = self._data.copy()

    def make_treatment_effective(self, x):
        self._df[self._treatment_names] = x

    def disrupt_causes(self):
        pass

    def point_sample(self):
            if self.num_cores == 1:
                sampled_outcomes = self._df[self._treatment_names +
                                            self._target_estimand.backdoor_variables].apply(self._sample_point, axis=1)
            else:
                from multiprocessing import Pool
                p = Pool(self.num_cores)
                sampled_outcomes = np.array(p.map(self.sampler.sample_point,
                                            self._df[self._treatment_names +
                                                     self._target_estimand.backdoor_variables].values))
                sampled_outcomes = pd.DataFrame(sampled_outcomes, columns=self._outcome_names)
            self._df[self._outcome_names] = sampled_outcomes

    def sample(self):
        sampled_outcomes = self.sampler.sample(self._df[self._treatment_names +
                                                     self._target_estimand.backdoor_variables].values)
        sampled_outcomes = pd.DataFrame(sampled_outcomes, columns=self._outcome_names)
        self._df[self._outcome_names] = sampled_outcomes

    def do_sample(self, x):
        self.reset()
        self.disrupt_causes()
        self.make_treatment_effective(x)
        if self.point_sampler:
            self.point_sample()
        else:
            self.sample()
        return self._df
