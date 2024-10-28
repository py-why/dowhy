import numpy as np
import pandas as pd

from dowhy.utils.dgp import DataGeneratingProcess


class QuadraticDataGeneratingProcess(DataGeneratingProcess):
    """
    Implements a data generating process that returns data having quadratic relationship between the treatment, outcome and confounders
    """

    POWER = 2
    NAME = "Quadratic DGP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_gen = False

        if self.weights == {} and self.bias == {}:
            self.auto_gen = True

    def generate_data(self, sample_size):
        self.weights = {}
        self.bias = {}
        weights = []
        bias = []
        treatment = []
        outcome = []
        y_treatment = []
        y_control = []
        confounder = np.random.randn(sample_size, len(self.confounder))
        effect_modifier = np.random.randn(sample_size, len(self.effect_modifier))
        control_value = np.zeros((sample_size, len(self.treatment)))
        treatment_value = np.ones((sample_size, len(self.treatment)))

        if self.auto_gen:
            self.generation_process()
        treatment.append(
            np.matmul(confounder, self.weights["confounder=>treatment"])
            + np.random.randn(sample_size, len(self.treatment))
            + self.bias["confounder=>treatment"]
        )

        for i in range(QuadraticDataGeneratingProcess.POWER):
            outcome.append(
                np.matmul(confounder, self.weights["confounder=>outcome"])
                + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
                + np.matmul(treatment[0], self.weights["treatment=>outcome"])
                + self.bias["confounder=>outcome"]
            )
            y_control.append(
                np.matmul(confounder, self.weights["confounder=>outcome"])
                + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
                + np.matmul(control_value, self.weights["treatment=>outcome"])
                + self.bias["confounder=>outcome"]
            )
            y_treatment.append(
                np.matmul(confounder, self.weights["confounder=>outcome"])
                + np.matmul(effect_modifier, self.weights["effect_modifier=>outcome"])
                + np.matmul(treatment_value, self.weights["treatment=>outcome"])
                + self.bias["confounder=>outcome"]
            )

            weights.append(self.weights)
            bias.append(self.bias)
            if self.auto_gen:
                self.generation_process()

        treatment = treatment[0]  # treatment[1]
        if self.treatment_is_binary:
            treatment = self.convert_to_binary(treatment)

        outcome = outcome[0] * outcome[1]
        y_control = y_control[0] * y_control[1]
        y_treatment = y_treatment[0] * y_treatment[1]
        self.true_value = np.mean(y_treatment - y_control, axis=0)

        self.weights = weights
        self.bias = bias

        return pd.DataFrame(
            np.hstack((effect_modifier, confounder, treatment, outcome)),
            columns=self.effect_modifier + self.confounder + self.treatment + self.outcome,
        )

    def generation_process(self):
        self.weights["confounder=>treatment"] = self.generate_weights((len(self.confounder), len(self.treatment)))
        self.weights["confounder=>outcome"] = self.generate_weights((len(self.confounder), len(self.outcome)))
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
        Quadratic Data Generating Process
        ----------------------------------
        """

        rep = header + rep

        return rep
