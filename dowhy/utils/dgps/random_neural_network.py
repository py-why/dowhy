import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from dowhy.utils.dgp import DataGeneratingProcess


class RandomNeuralNetwork(DataGeneratingProcess):

    TRAINING_SAMPLE_SIZE = 10
    RANDOM_STATE = 0
    DEFAULT_ARCH = (50, 50, 50)
    DEFAULT_ARCH_DICT = {
        "confounder=>treatment": DEFAULT_ARCH,
        "confounder=>outcome": DEFAULT_ARCH,
        "effect_modifier=>outcome": DEFAULT_ARCH,
        "treatment=>outcome": DEFAULT_ARCH,
    }
    NAME = "Random Neural Network"

    def __init__(self, **kwargs):
        """
        Understanding Architectures in MLP Regressor
        https://stackoverflow.com/questions/35363530/python-scikit-learn-mlpclassifier-hidden-layer-sizes

        More Information about Random State
        https://stackoverflow.com/questions/42191717/python-random-state-in-splitting-dataset/42197534
        https://stats.stackexchange.com/questions/80407/am-i-creating-bias-by-using-the-same-random-seed-over-and-over
        """
        super().__init__(**kwargs)
        self.arch = kwargs.pop("arch", RandomNeuralNetwork.DEFAULT_ARCH_DICT)
        self.random_state = kwargs.pop("random_state", RandomNeuralNetwork.RANDOM_STATE)
        self.nn = {}
        self.nn["confounder=>treatment"] = MLPRegressor(
            random_state=self.random_state, hidden_layer_sizes=self.arch["confounder=>treatment"]
        )
        self.nn["confounder=>outcome"] = MLPRegressor(
            random_state=self.random_state, hidden_layer_sizes=self.arch["confounder=>outcome"]
        )
        self.nn["effect_modifier=>outcome"] = MLPRegressor(
            random_state=self.random_state, hidden_layer_sizes=self.arch["effect_modifier=>outcome"]
        )
        self.nn["treatment=>outcome"] = MLPRegressor(
            random_state=self.random_state, hidden_layer_sizes=self.arch["treatment=>outcome"]
        )

    def generate_data(self, sample_size):
        self.generation_process()
        confounder = np.random.randn(sample_size, len(self.confounder))
        effect_modifier = np.random.randn(sample_size, len(self.effect_modifier))
        control_value = np.zeros((sample_size, len(self.treatment)))
        treatment_value = np.ones((sample_size, len(self.treatment)))

        treatment = self.nn["confounder=>treatment"].predict(confounder)
        treatment = treatment[:, np.newaxis]
        if self.treatment_is_binary:
            treatment = self.convert_to_binary(treatment)

        if treatment.ndim == 1:
            treatment = np.reshape(treatment, (-1, 1))

        outcome = (
            self.nn["confounder=>outcome"].predict(np.hstack((confounder, effect_modifier)))
            + self.nn["effect_modifier=>outcome"].predict(effect_modifier)
            + self.nn["treatment=>outcome"].predict(np.hstack((confounder, effect_modifier, treatment)))
        )
        y_control = (
            self.nn["confounder=>outcome"].predict(np.hstack((confounder, effect_modifier)))
            + self.nn["effect_modifier=>outcome"].predict(effect_modifier)
            + self.nn["treatment=>outcome"].predict(np.hstack((confounder, effect_modifier, control_value)))
        )
        y_treatment = (
            self.nn["confounder=>outcome"].predict(np.hstack((confounder, effect_modifier)))
            + self.nn["effect_modifier=>outcome"].predict(effect_modifier)
            + self.nn["treatment=>outcome"].predict(np.hstack((confounder, effect_modifier, treatment_value)))
        )

        # Understanding Neural Network weights
        # Refer to this link:https://stackoverflow.com/questions/50937628/mlp-classifier-neurons-weights
        self.weights = {key: self.nn[key].coefs_ for key in self.nn.keys()}
        self.bias = {key: self.nn[key].intercepts_ for key in self.nn.keys()}

        if outcome.ndim == 1:
            outcome = np.reshape(outcome, (-1, 1))
        if y_control.ndim == 1:
            y_control = np.reshape(y_control, (-1, 1))
        if y_treatment.ndim == 1:
            y_treatment = np.reshape(y_treatment, (-1, 1))
        self.true_value = np.mean(y_treatment - y_control, axis=0)

        return pd.DataFrame(
            np.hstack((effect_modifier, confounder, treatment, outcome)),
            columns=self.effect_modifier + self.confounder + self.treatment + self.outcome,
        )

    def generation_process(self):
        X = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.confounder))
        y = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.treatment))
        self.nn["confounder=>treatment"].fit(X, y)

        X = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.confounder) + len(self.effect_modifier))
        y = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.outcome))
        self.nn["confounder=>outcome"].fit(X, y)

        X = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.effect_modifier))
        y = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.outcome))
        self.nn["effect_modifier=>outcome"].fit(X, y)

        X = np.random.randn(
            RandomNeuralNetwork.TRAINING_SAMPLE_SIZE,
            len(self.confounder) + len(self.effect_modifier) + len(self.treatment),
        )
        y = np.random.randn(RandomNeuralNetwork.TRAINING_SAMPLE_SIZE, len(self.outcome))
        self.nn["treatment=>outcome"].fit(X, y)

    def __str__(self):
        rep = super().__str__()

        header = """
        Random Neural Network Data Generating Process
        ---------------------------------------------
        """
        rep += """
        arch:{}
        nn:{}
        random_state:{}
        """.format(
            self.arch, self.nn, self.random_state
        )

        rep = header + rep

        return rep
