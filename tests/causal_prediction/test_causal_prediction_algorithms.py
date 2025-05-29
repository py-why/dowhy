import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset

from dowhy.causal_prediction.algorithms.cacm import CACM
from dowhy.causal_prediction.algorithms.erm import ERM
from dowhy.causal_prediction.dataloaders.get_data_loader import get_loaders
from dowhy.causal_prediction.models.networks import MLP, Classifier
from dowhy.datasets import linear_dataset


class LinearTensorDataset:
    N_WORKERS = 0

    def __init__(self, n_envs, n_samples, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.datasets = []

        for env in range(n_envs):
            data = linear_dataset(
                beta=10,
                num_common_causes=2,
                num_instruments=0,
                num_samples=n_samples,
                treatment_is_binary=True,
                outcome_is_binary=True,
            )
            df = data["df"]

            # Use treatment as input, outcome as label
            x = torch.tensor(df[data["treatment_name"]].values, dtype=torch.float32).reshape(-1, 1)
            y = torch.tensor(df[data["outcome_name"]].values, dtype=torch.long)

            # Use common causes as attributes
            cc_names = data["common_causes_names"]
            a = torch.tensor(df[cc_names].values, dtype=torch.float32)
            self.datasets.append(TensorDataset(x, y, a))

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


@pytest.mark.usefixtures("fixed_seed")
@pytest.mark.parametrize(
    "algorithm_cls, algorithm_kwargs",
    [
        (ERM, {}),
        (CACM, {"gamma": 1e-2, "attr_types": ["causal"], "lambda_causal": 10.0}),
    ],
)
def test_causal_prediction_training_and_eval(algorithm_cls, algorithm_kwargs, fixed_seed):
    # Use the new linear dataset-based class
    dataset = LinearTensorDataset(n_envs=4, n_samples=1000, input_shape=(1,), num_classes=2)
    loaders = get_loaders(dataset, train_envs=[0, 1], batch_size=64, val_envs=[2], test_envs=[3])

    # Model
    n_inputs = dataset.input_shape[0]
    mlp_width = 128
    mlp_depth = 4
    mlp_dropout = 0.1
    n_outputs = mlp_width
    featurizer = MLP(n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout)
    classifier = Classifier(featurizer.n_outputs, dataset.num_classes)
    model = torch.nn.Sequential(featurizer, classifier)

    # Train
    algorithm = algorithm_cls(model, lr=1e-3, **algorithm_kwargs)
    trainer = pl.Trainer(devices=1, max_epochs=5, accelerator="cpu", logger=False, enable_checkpointing=False)

    # Fit
    trainer.fit(algorithm, loaders["train_loaders"], loaders["val_loaders"])

    # Check results
    results = trainer.test(algorithm, dataloaders=loaders["test_loaders"])
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        if "test_acc" in r:
            assert r["test_acc"] > 0.7, f"Test accuracy too low: {r['test_acc']}"
        if "test_loss" in r:
            assert r["test_loss"] < 1.0, f"Test loss too high: {r['test_loss']}"
