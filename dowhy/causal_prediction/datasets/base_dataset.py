# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
`MultipleDomainDataset` class in this file is borrowed from DomainBed: https://github.com/facebookresearch/DomainBed
    @inproceedings{gulrajani2021in,
     title={In Search of Lost Domain Generalization},
     author={Ishaan Gulrajani and David Lopez-Paz},
     booktitle={International Conference on Learning Representations},
     year={2021},
    }
"""


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
