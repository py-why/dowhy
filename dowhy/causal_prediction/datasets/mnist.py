# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torchvision
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate

from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset

""" MNIST Causal, Independent and Causal+Independent datasets

    The class structure for datasets is adapted from OoD-Bench:
        @inproceedings{ye2022ood,
         title={OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization},
         author={Ye, Nanyang and Li, Kaican and Bai, Haoyue and Yu, Runpeng and Hong, Lanqing and Zhou, Fengwei and Li, Zhenguo and Zhu, Jun},
         booktitle={CVPR},
         year={2022}
        }

    * datasets initialized from torchvision.datasets.MNIST
    * We assume causal attribute (Acause) = color, independent attribute (Aind) = rotation
    * Environments/domains stored in list self.datasets (required for all datasets)
    * Default env structure is TensorDataset(x, y, a)
        * a is a combined tensor for all attributes (metadata) a1, a2, ..., ak
        * a's shape is (n, k) where n is the number of samples in the environment
"""


# single-attribute Causal
class MNISTCausalAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%", "+80%", "-90%", "-90%"]
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, download=True):
        """Class for MNISTCausalAttribute dataset.

        :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
        :param download: Binary flag indicating whether data should be downloaded
        :returns: an instance of MultipleDomainDataset class

        """

        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = (0.1, 0.2, 0.9)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_dataset(images, labels, environments[-1]))

        # test environment
        original_dataset_te = MNIST(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(self.color_dataset(original_images, original_labels, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        """
        Transform MNIST dataset to introduce correlation between attribute (color) and label.
        There is a direct-causal relationship between label Y and color.

        :param images: original MNIST images
        :param labels: original MNIST labels
        :param environment: Value of correlation between color and label
        :returns: TensorDataset containing transformed images, labels, and attributes (color)
        """

        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability environment
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        a = torch.unsqueeze(colors, 1)

        return TensorDataset(x, y, a)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


# single-attribute Independent
class MNISTIndAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["15", "60", "90", "90"]
    INPUT_SHAPE = (1, 14, 14)

    def __init__(self, root, download=True):
        """Class for MNISTIndAttribute dataset.

        :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
        :param download: Binary flag indicating whether data should be downloaded
        :returns: an instance of MultipleDomainDataset class

        """

        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        angles = ["15", "60", "90"]
        for i, env in enumerate(angles[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.rotate_dataset(images, labels, i, angles[i]))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.rotate_dataset(images, labels, len(angles) - 1, angles[-1]))

        # test environment
        original_dataset_te = MNIST(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(self.rotate_dataset(original_images, original_labels, len(angles) - 1, angles[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def rotate_dataset(self, images, labels, env_id, angle):
        """
        Transform MNIST dataset by applying rotation to images.
        Attribute (rotation angle) is independent of label Y.

        :param images: original MNIST images
        :param labels: original MNIST labels
        :param angle: Value of rotation angle used for transforming the image
        :returns: TensorDataset containing transformed images, labels, and attributes (angle)
        """
        rotation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(
                    lambda x: rotate(
                        x, int(angle), fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    )
                ),
                transforms.ToTensor(),
            ]
        )

        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        x = torch.zeros(len(images), 1, 14, 14)
        for i in range(len(images)):
            x[i] = rotation(images[i].float().div_(255.0))

        y = labels.view(-1).long()
        a = torch.full((y.shape[0],), env_id, dtype=torch.float32)
        a = torch.unsqueeze(a, 1)

        return TensorDataset(x, y, a)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


# multi-attribute Causal + Independent
class MNISTCausalIndAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%, 15", "+80%, 16", "-90%, 90", "-90%, 90"]
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, download=True):
        """Class for MNISTIndAttribute dataset.

        :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
        :param download: Binary flag indicating whether data should be downloaded
        :returns: an instance of MultipleDomainDataset class

        """

        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environments = (0.1, 0.2, 0.9)
        angles = ["15", "60", "90"]
        for i, env in enumerate(environments[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_rot_dataset(images, labels, env, i, angles[i]))
        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(self.color_rot_dataset(images, labels, environments[-1], len(angles) - 1, angles[-1]))

        # test environment
        original_dataset_te = MNIST(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(
            self.color_rot_dataset(original_images, original_labels, environments[-1], len(angles) - 1, angles[-1])
        )

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

    def color_rot_dataset(self, images, labels, environment, env_id, angle):
        """
        Transform MNIST dataset by (i) applying rotation to images, then (ii) introducing correlation between attribute (color) and label.
        Attribute (rotation angle) is independent of label Y; there is a direct-causal relationship between label Y and color.

        :param images: original MNIST images
        :param labels: original MNIST labels
        :param environment: Value of correlation between color and label
        :param angle: Value of rotation angle used for transforming the image
        :returns: TensorDataset containing transformed images, labels, and attributes (color, angle)
        """
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # rotate the image by angle in parameter
        images = self.rotate_dataset(images, angle)

        images, labels, colors = self.color_dataset(images, labels, environment)

        x = images  # .float().div_(255.0)
        y = labels.view(-1).long()
        angles = torch.full((y.shape[0],), env_id, dtype=torch.float32)
        a = torch.stack((colors, angles), 1)

        return TensorDataset(x, y, a)

    def color_dataset(self, images, labels, environment):
        """
        Transform MNIST dataset to introduce correlation between attribute (color) and label.
        There is a direct-causal relationship between label Y and color.

        :param images: rotated MNIST images
        :param labels: original MNIST labels
        :param environment: Value of correlation between color and label
        :returns: transformed images, labels, and attributes (color)
        """

        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability environment
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        return images, labels, colors

    def rotate_dataset(self, images, angle):
        """
        Transform MNIST dataset by applying rotation to images.
        Attribute (rotation angle) is independent of label Y.

        :param images: original MNIST images
        :param angle: Value of rotation angle used for transforming the image
        :returns: transformed images
        """
        rotation = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: transforms.functional.rotate(x, int(angle), fill=(0,))),
                transforms.ToTensor(),
            ]
        )

        x = torch.zeros(len(images), 14, 14)
        for i in range(len(images)):
            x[i] = rotation(images[i].float().div_(255.0))
        return x

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
