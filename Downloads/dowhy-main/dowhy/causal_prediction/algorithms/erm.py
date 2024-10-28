import torch
from torch.nn import functional as F

from dowhy.causal_prediction.algorithms.base_algorithm import PredictionAlgorithm


class ERM(PredictionAlgorithm):
    def __init__(self, model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9):
        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)

        """Class for ERM Algorithm.

        :param model: Networks used for training. `model` type expected is torch.nn.Sequential(featurizer, classifier) where featurizer and classifier are of type torch.nn.Module.
        :param optimizer: Optimization algorithm used for training. Currently supports "Adam" and "SGD".
        :param lr: learning rate for ERM
        :param weight_decay: Value of weight decay for optimizer
        :param betas: Adam configuration parameters (beta1, beta2), exponential decay rate for the first moment and second-moment estimates, respectively.
        :param momentum: Value of momentum for SGD optimzer
        :returns: an instance of PredictionAlgorithm class

        """

    def training_step(self, train_batch, batch_idx):
        """
        Override `training_step` from PredictionAlgorithm class for ERM-specific training loop.

        """

        x = torch.cat([x for x, y, _ in train_batch])
        y = torch.cat([y for x, y, _ in train_batch])

        out = self.model(x)
        loss = F.cross_entropy(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss
