from dowhy.do_sampler import DoSampler
from dowhy.utils.propensity_score import propensity_score


class WeightingSampler(DoSampler):
    def __init__(self, data, identified_estimand, treatments, outcomes, *args, params=None, variable_types=None,
                 num_cores=1, keep_original_treatment=False, **kwargs):
        """
        g, df, data_types

        """
        super().__init__(data, identified_estimand, treatments, outcomes, params=params, variable_types=variable_types,
                 num_cores=num_cores, keep_original_treatment=keep_original_treatment)

        self.logger.info("Using WeightingSampler for do sampling.")
        self.logger.info("Caution: do samplers assume iid data.")
        if variable_types[self._treatment_names[0]] != 'b' or len(self._treatment_names) > 1:
            raise Exception("WeightingSampler only supports atomic binary treatments")
        self.point_sampler = False

    def make_treatment_effective(self, x):
        to_sample = self._df.copy()
        if not self.keep_original_treatment:
            for treatment, value in x.items():
                to_sample = to_sample[to_sample[treatment] == value]
        self._df = to_sample

    def disrupt_causes(self):
        self._df['propensity_score'] = propensity_score(self._data[self._target_estimand.backdoor_variables],
                                                        self._data[self._treatment_names])
        self._df['weight'] = self.compute_weights()

    def sample(self):
        self._df = self._df.sample(len(self._data), replace=True, weights=self._df['weight'])
        self._df.index = self._data.index

    def compute_weights(self):
        weights = self._df[self._treatment_names[0]] / self._df['propensity_score'] \
               + (1. - self._df[self._treatment_names[0]]) / (1. - self._df['propensity_score'])
        return weights
