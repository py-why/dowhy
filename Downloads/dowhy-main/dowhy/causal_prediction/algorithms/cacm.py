import torch
from torch.nn import functional as F

from dowhy.causal_prediction.algorithms.base_algorithm import PredictionAlgorithm
from dowhy.causal_prediction.algorithms.regularization import Regularizer


class CACM(PredictionAlgorithm):
    def __init__(
        self,
        model,
        optimizer="Adam",
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        momentum=0.9,
        kernel_type="gaussian",
        ci_test="mmd",
        attr_types=[],
        E_conditioned=True,
        E_eq_A=[],
        gamma=1e-6,
        lambda_causal=1.0,
        lambda_conf=1.0,
        lambda_ind=1.0,
        lambda_sel=1.0,
    ):
        """Class for Causally Adaptive Constraint Minimization (CACM) Algorithm.
            @article{Kaur2022ModelingTD,
             title={Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization},
             author={Jivat Neet Kaur and Emre Kıcıman and Amit Sharma},
             journal={ArXiv},
             year={2022},
             volume={abs/2206.07837},
             url={https://arxiv.org/abs/2206.07837}
            }

        :param model: Networks used for training. `model` type expected is torch.nn.Sequential(featurizer, classifier) where featurizer and classifier are of type torch.nn.Module.
        :param optimizer: Optimization algorithm used for training. Currently supports "Adam" and "SGD".
        :param lr: learning rate for CACM
        :param weight_decay: Value of weight decay for optimizer
        :param betas: Adam configuration parameters (beta1, beta2), exponential decay rate for the first moment and second-moment estimates, respectively.
        :param momentum: Value of momentum for SGD optimzer
        :param kernel_type: Kernel type for MMD penalty. Currently, supports "gaussian" (RBF). If None, distance between mean and second-order statistics (covariances) is used.
        :param ci_test: Conditional independence metric used for regularization penalty. Currently, MMD is supported.
        :param attr_types: list of attribute types (based on relationship with label Y); should be ordered according to attribute order in loaded dataset.
            Currently, 'causal' (Causal), 'conf' (Confounded), 'ind' (Independent) and 'sel' (Selected) are supported.
            For single-shift datasets, use: ['causal'], ['ind']
            For multi-shift datasets, use: ['causal', 'ind']
        :param E_conditioned: Binary flag indicating whether E-conditioned regularization has to be applied
        :param E_eq_A: list indicating indices of attributes that coincide with environment (E) definition; default is empty.
        :param gamma: kernel bandwidth for MMD (due to implementation, the kernel bandwdith will actually be the reciprocal of gamma i.e., gamma=1e-6 implies kernel bandwidth=1e6. See `mmd_compute` in utils.py)
        :param lambda_causal: MMD penalty hyperparameter for Causal shift
        :param lambda_conf: MMD penalty hyperparameter for Confounded shift
        :param lambda_ind: MMD penalty hyperparameter for Independent shift
        :param lambda_sel: MMD penalty hyperparameter for Selected shift
        :returns: an instance of PredictionAlgorithm class

        """

        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)

        self.CACMRegularizer = Regularizer(E_conditioned, ci_test, kernel_type, gamma)

        self.attr_types = attr_types
        self.E_eq_A = E_eq_A
        self.lambda_causal = lambda_causal
        self.lambda_conf = lambda_conf
        self.lambda_ind = lambda_ind
        self.lambda_sel = lambda_sel

    def training_step(self, train_batch, batch_idx):
        """
        Override `training_step` from PredictionAlgorithm class for CACM-specific training loop.

        """

        self.featurizer = self.model[0]
        self.classifier = self.model[1]

        minibatches = train_batch

        objective = 0
        correct, total = 0, 0
        penalty_causal, penalty_conf, penalty_ind, penalty_sel = 0, 0, 0, 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi, _ in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            correct += (torch.argmax(classifs[i], dim=1) == targets[i]).float().sum().item()
            total += classifs[i].shape[0]

        objective /= nmb
        loss = objective

        if self.attr_types != []:
            for attr_type_idx, attr_type in enumerate(self.attr_types):
                attribute_labels = [
                    ai for _, _, ai in minibatches
                ]  # [(batch_size, num_attrs)_1, batch_size, num_attrs)_2, ..., (batch_size, num_attrs)_(num_environments)]

                E_eq_A_attr = attr_type_idx in self.E_eq_A

                # Acause regularization
                if attr_type == "causal":
                    penalty_causal += self.CACMRegularizer.conditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], [targets], nmb, E_eq_A_attr
                    )

                # Aconf regularization
                elif attr_type == "conf":
                    penalty_conf += self.CACMRegularizer.unconditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], nmb, E_eq_A_attr
                    )

                # Aind regularization
                elif attr_type == "ind":
                    penalty_ind += self.CACMRegularizer.unconditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], nmb, E_eq_A_attr
                    )

                # Asel regularization
                elif attr_type == "sel":
                    penalty_sel += self.CACMRegularizer.conditional_reg(
                        classifs, [a[:, attr_type_idx] for a in attribute_labels], [targets], nmb, E_eq_A_attr
                    )

            if nmb > 1:
                penalty_causal /= nmb * (nmb - 1) / 2
                penalty_conf /= nmb * (nmb - 1) / 2
                penalty_ind /= nmb * (nmb - 1) / 2
                penalty_sel /= nmb * (nmb - 1) / 2

            # Compile loss
            loss += self.lambda_causal * penalty_causal
            loss += self.lambda_conf * penalty_conf
            loss += self.lambda_ind * penalty_ind
            loss += self.lambda_sel * penalty_sel

            if torch.is_tensor(penalty_causal):
                penalty_causal = penalty_causal.item()
                self.log("penalty_causal", penalty_causal, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_conf):
                penalty_conf = penalty_conf.item()
                self.log("penalty_conf", penalty_conf, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_ind):
                penalty_ind = penalty_ind.item()
                self.log("penalty_ind", penalty_ind, on_step=False, on_epoch=True, prog_bar=True)
            if torch.is_tensor(penalty_sel):
                penalty_sel = penalty_sel.item()
                self.log("penalty_sel", penalty_sel, on_step=False, on_epoch=True, prog_bar=True)

        elif self.graph is not None:
            pass  # TODO

        else:
            raise ValueError("No attribute types or graph provided.")

        acc = correct / total

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss
