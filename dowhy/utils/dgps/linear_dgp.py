import numpy as np
import pandas as pd

from dowhy.utils.dgp import DataGeneratingProcess


class LinearDataGeneratingProcess(DataGeneratingProcess):
    """
    Implements a data generating process that returns data having linear relationship between the treatment, outcome and confounders
    """

    NAME = "Linear DGP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_gen = False

        if self.weights == {} and self.bias == {}:
            self.auto_gen = True

    def generate_data(self, sample_size):
        if self.auto_gen:
            self.generation_process()

        control_value = np.zeros((sample_size, len(self.treatment)))
        treatment_value = np.ones((sample_size, len(self.treatment)))
        confounder = np.random.randn(sample_size, len(self.confounder))
        effect_modifier = np.zeros((sample_size, len(self.effect_modifier)))  # random.randn

        treatment = (
            np.matmul(confounder, self.weights["confounder=>treatment"])
            + np.random.randn(sample_size, len(self.treatment))
            + self.bias["confounder=>treatment"]
        )
        if self.treatment_is_binary:
            treatment = self.convert_to_binary(treatment)

        outcome = (
            np.matmul(confounder, self.weights["confounder=>outcome"])
            + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
            + np.matmul(treatment, self.weights["treatment=>outcome"])
            + self.bias["confounder=>outcome"]
        )
        y_control = (
            np.matmul(confounder, self.weights["confounder=>outcome"])
            + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
            + np.matmul(control_value, self.weights["treatment=>outcome"])
            + self.bias["confounder=>outcome"]
        )
        y_treatment = (
            np.matmul(confounder, self.weights["confounder=>outcome"])
            + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
            + np.matmul(treatment_value, self.weights["treatment=>outcome"])
            + self.bias["confounder=>outcome"]
        )
        self.true_value = np.mean(y_treatment - y_control, axis=0)

        return pd.DataFrame(
            np.hstack((effect_modifier, confounder, treatment, outcome)),
            columns=self.effect_modifier + self.confounder + self.treatment + self.outcome,
        )

    def generation_process(self):
        self.weights["confounder=>treatment"] = self.generate_weights((len(self.confounder), len(self.treatment)))
        self.weights["confounder=>treatment"][0,] = (
            self.weights["confounder=>treatment"][0,] + 100
        )  # increasing weight of the first confounder
        self.weights["confounder=>outcome"] = self.generate_weights((len(self.confounder), len(self.outcome)))
        self.weights["confounder=>outcome"][0,] = self.weights["confounder=>outcome"][0,] + 100
        self.weights["effect_modifier=>outcome"] = self.generate_weights((len(self.effect_modifier), len(self.outcome)))
        self.weights["treatment=>outcome"] = self.generate_weights((len(self.treatment), len(self.outcome)))

        self.bias["confounder=>treatment"] = self.generate_bias(len(self.treatment))
        self.bias["confounder=>outcome"] = self.generate_bias(len(self.outcome))

    def generate_weights(self, dimensions):
        return np.random.randn(*dimensions)

    def generate_bias(self, dimensions):
        return np.random.randn(dimensions)

    def __str__(self):
        rep = super().__str__()

        header = """
        Linear Data Generating Process
        -------------------------------
        """

        rep = header + rep

        return rep
