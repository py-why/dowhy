import pytest
import pytorch_lightning as pl
import torch

from dowhy.causal_prediction.algorithms.cacm import CACM
from dowhy.causal_prediction.algorithms.erm import ERM
from dowhy.causal_prediction.dataloaders.get_data_loader import get_loaders
from dowhy.causal_prediction.models.networks import MLP, Classifier
from dowhy.datasets import linear_dataset


class LinearTensorDataset:
    N_WORKERS = 0

    def __init__(self, env_specs, n_samples, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.datasets = []
        self.env_names = []

        for env_name, beta in env_specs:
            data = linear_dataset(
                beta=beta,
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
            self.datasets.append(torch.utils.data.TensorDataset(x, y, a))
            self.env_names.append(env_name)

        self.env_by_name = {name: idx for idx, name in enumerate(self.env_names)}

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


def _evaluate_algorithm(algorithm_cls, algorithm_kwargs):
    # Setup dataset
    env_specs = [
        ("train", 10),
        ("val_same_distribution", 10),
        ("test_new_distribution", 5),
    ]
    dataset = LinearTensorDataset(env_specs=env_specs, n_samples=1000, input_shape=(1,), num_classes=2)
    envs = dataset.env_by_name

    loaders = get_loaders(
        dataset,
        train_envs=[envs["train"]],
        val_envs=[envs["val_same_distribution"]],
        test_envs=[envs["test_new_distribution"]],
        batch_size=64,
    )

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
    trainer.fit(algorithm, loaders["train_loaders"], loaders["val_loaders"])

    # Test
    val_same_distribution = trainer.test(algorithm, dataloaders=loaders["val_loaders"], verbose=False)[0]["test_acc"]
    test_new_distribution = trainer.test(algorithm, dataloaders=loaders["test_loaders"], verbose=False)[0]["test_acc"]

    return val_same_distribution, test_new_distribution


@pytest.mark.usefixtures("fixed_seed")
def test_cacm_vs_erm_accuracy_and_gap():
    # CACM
    val_same_distribution_cacm, test_new_distribution_cacm = _evaluate_algorithm(
        CACM,
        {
            "gamma": 1e-4,
            "attr_types": ["causal"],
            "lambda_causal": 1.0,
        },
    )
    gap_cacm = val_same_distribution_cacm - test_new_distribution_cacm

    # ERM
    val_same_distribution_erm, test_new_distribution_erm = _evaluate_algorithm(ERM, {})
    gap_erm = val_same_distribution_erm - test_new_distribution_erm

    # Accuracy checks
    assert val_same_distribution_erm > 0.7
    assert test_new_distribution_erm > 0.7
    assert val_same_distribution_cacm > 0.7
    assert test_new_distribution_cacm > 0.7

    # Generalization gap check
    assert gap_erm > gap_cacm, f"Expected ERM to degrade more. ERM gap={gap_erm:.4f}, CACM gap={gap_cacm:.4f}"
