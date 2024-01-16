import numpy as np


class DataGeneratingProcess:

    DEFAULT_PERCENTILE = 0.9

    def __init__(self, **kwargs):
        """
        Base class for implementation of data generating process.

        Subclasses implement functions that create various data generating processes. All data generating processes are in the package "dowhy.utils.dgps".
        """
        self.treatment = kwargs["treatment"]
        self.outcome = kwargs["outcome"]
        self.confounder = kwargs["confounder"]
        self.effect_modifier = kwargs["effect_modifier"]
        self.weights = kwargs.pop("weights", {})
        self.bias = kwargs.pop("bias", {})
        self.seed = kwargs.pop("seed", None)
        self.treatment_is_binary = kwargs.pop("treatment_is_binary", False)
        if self.treatment_is_binary:
            self.percentile = kwargs.pop("percentile", DataGeneratingProcess.DEFAULT_PERCENTILE)
        elif kwargs.pop("percentile", None) is not None:
            raise ValueError("Cannot use percentile, if the input is non-binary")
        else:
            self.percentile = "NA"
        self.true_value = None
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_data(self):
        raise NotImplementedError()

    def generation_process(self):
        raise NotImplementedError()

    def convert_to_binary(self, data, deterministic=False):
        if deterministic:
            precentile = np.percentile(data, self.percentile, axis=0)
            binary_treat_value = data <= precentile
        else:
            temp = data.argsort(axis=0)
            ranks = np.empty_like(temp)
            ranks[temp[:, 0], 0] = np.arange(data.shape[0])
            prob_t = ranks / data.shape[0]
            # Generating data with equal 0 and 1 (since ranks are uniformly distributed)
            binary_treat_value = np.random.binomial(1, prob_t[:, 0], data.shape[0])

            # Flipping some values
            if self.percentile >= 0.5:
                mask = np.random.binomial(
                    1, (1 - self.percentile) * 2, len(binary_treat_value[binary_treat_value == 1])
                )
                binary_treat_value[binary_treat_value == 1] = mask * binary_treat_value[binary_treat_value == 1]
            else:
                mask = np.random.binomial(1, 1 - self.percentile * 2, len(binary_treat_value[binary_treat_value == 0]))
                binary_treat_value[binary_treat_value == 0] = mask + binary_treat_value[binary_treat_value == 0]
            binary_treat_value = binary_treat_value[:, np.newaxis]

        return binary_treat_value.astype(float)

    def __str__(self):
        rep = """
        treatment:{}
        outcome:{}
        confounder: {}
        effect_modifier: {}
        weights: {}
        bias: {}
        seed: {}
        treatment_is_binary: {}
        percentile: {}
        """.format(
            self.treatment,
            self.outcome,
            self.confounder,
            self.effect_modifier,
            self.weights,
            self.bias,
            self.seed,
            self.treatment_is_binary,
            self.percentile,
        )

        return rep
