from dowhy.causal_prediction.dataloaders import misc
from dowhy.causal_prediction.dataloaders.fast_data_loader import FastDataLoader, InfiniteDataLoader


def get_train_loader(dataset, envs, batch_size, class_balanced=False):
    """Return training dataloaders.

    :param dataset: dataset class containing list of environments
    :param envs: list containing indices of training domains in the dataset
    :param batch_size: Value of batch size to be used for dataloaders
    :param class_balanced: Binary flag indicating whether balanced sampling is to be done between classes

    :returns: list of dataloaders
    """
    splits = []

    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            if class_balanced:
                weights = misc.make_weights_for_balanced_classes(env)
            else:
                weights = None
            splits.append((env, weights))

    train_loaders = [
        InfiniteDataLoader(dataset=env, weights=env_weights, batch_size=batch_size, num_workers=dataset.N_WORKERS)
        for env, env_weights in splits
    ]

    return train_loaders


def get_eval_loader(dataset, envs, batch_size, class_balanced=False):
    """Return evaluation dataloaders (test/validation).

    :param dataset: dataset class containing list of environments
    :param envs: list containing indices of validation/test domains in the dataset
    :param batch_size: Value of batch size to be used for dataloaders
    :param class_balanced: Binary flag indicating whether balanced sampling is to be done between classes

    :returns: list of dataloaders
    """
    splits = []

    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            if class_balanced:
                weights = misc.make_weights_for_balanced_classes(env)
            else:
                weights = None
            splits.append((env, weights))

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=batch_size, num_workers=dataset.N_WORKERS) for env, _ in splits
    ]

    return eval_loaders


def get_train_eval_loader(dataset, envs, batch_size, class_balanced, holdout_fraction, trial_seed):
    """Return training and validation dataloaders.

    :param dataset: dataset class containing list of environments
    :param envs: list containing indices of training domains in the dataset
    :param batch_size: Value of batch size to be used for dataloaders
    :param class_balanced: Binary flag indicating whether balanced sampling is to be done between classes
    :param holdout_fraction: fraction of training data used for creating validation domains
    :param trial_seed: seed used for generating validation split from training data

    :returns: two lists of dataloaders for training (train_loaders) and validation (val_loaders) respectively
    """
    train_splits, val_splits = [], []

    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            val_, train_ = misc.split_dataset(
                env, int(len(env) * holdout_fraction), misc.seed_hash(trial_seed, env_idx)
            )

            if class_balanced:
                train_weights = misc.make_weights_for_balanced_classes(train_)
                val_weights = misc.make_weights_for_balanced_classes(val_)
            else:
                train_weights, val_weights = None, None

            train_splits.append((train_, train_weights))
            val_splits.append((val_, val_weights))

    train_loaders = [
        InfiniteDataLoader(dataset=env, weights=env_weights, batch_size=batch_size, num_workers=dataset.N_WORKERS)
        for env, env_weights in train_splits
    ]

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=batch_size, num_workers=dataset.N_WORKERS) for env, _ in val_splits
    ]

    return train_loaders, eval_loaders


def get_loaders(
    dataset,
    train_envs,
    batch_size,
    val_envs=None,
    test_envs=None,
    class_balanced=False,
    holdout_fraction=0.2,
    trial_seed=0,
):
    """Return training, validation, and test dataloaders.

    :param dataset: dataset class containing list of environments
    :param train_envs: list containing indices of training domains in the dataset
    :param batch_size: Value of batch size to be used for dataloaders
    :param val_envs: list containing indices of validation domains in the dataset. If None, fraction of training data (`holdout_fraction`) is used to create validation set.
    :param test_envs: list containing indices of test domains in the dataset
    :param class_balanced: Binary flag indicating whether balanced sampling is to be done between classes
    :param holdout_fraction: fraction of training data used for creating validation domains. This is used when `val_envs` is None.
    :param trial_seed: seed used for generating validation split from training data. This is used when `val_envs` is None.

    :returns: dictionary of list of dataloaders in the format
        {'train_loaders': [train_dataloader_1, train_dataloader_2, ....],
         'val_loaders': [val_dataloader_1, val_dataloader_2, ....],
         'test_loaders': [test_dataloader_1, test_dataloader_2, ....]
        }
    """

    loaders = {}

    if val_envs:  # use validation environment to initialize val_loaders
        loaders["train_loaders"] = get_train_loader(dataset, train_envs, batch_size, class_balanced)
        loaders["val_loaders"] = get_eval_loader(dataset, val_envs, batch_size, class_balanced)

    else:  # use subset of training data to initialize val_loaders
        loaders["train_loaders"], loaders["val_loaders"] = get_train_eval_loader(
            dataset, train_envs, batch_size, class_balanced, holdout_fraction, trial_seed
        )

    if test_envs:
        loaders["test_loaders"] = get_eval_loader(dataset, test_envs, batch_size, class_balanced)

    return loaders
