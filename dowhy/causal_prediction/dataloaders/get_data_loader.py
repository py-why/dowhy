from dowhy.causal_prediction.dataloaders import misc
from dowhy.causal_prediction.dataloaders.fast_data_loader import InfiniteDataLoader, FastDataLoader

def get_train_loader(dataset, envs, batch_size, 
    class_balanced=False):

    splits = []
    
    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            if class_balanced:
                weights = misc.make_weights_for_balanced_classes(env)
            else:
                weights = None
            splits.append((env, weights))
            
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS)
        for env, env_weights in splits]
    
    return train_loaders

def get_eval_loader(dataset, envs, batch_size, 
    class_balanced=False):

    splits = []
    
    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            if class_balanced:
                weights = misc.make_weights_for_balanced_classes(env)
            else:
                weights = None
            splits.append((env, weights))
            
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in splits]  
 
    return eval_loaders
 
def get_train_eval_loader(dataset, envs, batch_size,
        class_balanced, holdout_fraction, trial_seed):

    train_splits, val_splits = [], []

    for env_idx, env in enumerate(dataset):
        if env_idx in envs:
            val_, train_ = misc.split_dataset(env,
                int(len(env)*holdout_fraction),
                misc.seed_hash(trial_seed, env_idx))

            if class_balanced:
                train_weights = misc.make_weights_for_balanced_classes(train_)
                val_weights = misc.make_weights_for_balanced_classes(val_)
            else:
                train_weights, val_weights = None, None

            train_splits.append((train_, train_weights))
            val_splits.append((val_, val_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS)
        for env, env_weights in train_splits]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in val_splits]

    return train_loaders, eval_loaders

    
def get_loaders(dataset, train_envs, batch_size, val_envs=None, test_envs=None,
        class_balanced=False, holdout_fraction=0.2, trial_seed=0):
    
    loaders = {}
    
    if val_envs: # use validation environment to initialize val_loaders
        loaders['train_loaders'] = get_train_loader(dataset, train_envs, batch_size, class_balanced)
        loaders['val_loaders'] = get_eval_loader(dataset, val_envs, batch_size, class_balanced)
        
    else: # use subset of training data to initialize val_loaders
        loaders['train_loaders'], loaders['val_loaders'] = get_train_eval_loader(dataset, train_envs, batch_size, 
                                                                class_balanced, holdout_fraction, trial_seed)

    if test_envs:
        loaders['test_loaders'] = get_eval_loader(dataset, test_envs, batch_size, class_balanced)


    return loaders
        
