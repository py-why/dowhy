# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
The functions in this file are borrowed from DomainBed: https://github.com/facebookresearch/DomainBed
    @inproceedings{gulrajani2021in,
     title={In Search of Lost Domain Generalization},
     author={Ishaan Gulrajani and David Lopez-Paz},
     booktitle={International Conference on Learning Representations},
     year={2021},
    }
"""

import torch


def gaussian_kernel(x, y, gamma):
    return torch.exp(torch.cdist(x, y, p=2.0).pow(2).clamp_min_(1e-30).mul(-gamma))


def mmd_compute(x, y, kernel_type, gamma):
    if kernel_type == "gaussian":
        Kxx = gaussian_kernel(x, x, gamma).mean()
        Kyy = gaussian_kernel(y, y, gamma).mean()
        Kxy = gaussian_kernel(x, y, gamma).mean()
        return Kxx + Kyy - 2 * Kxy
    else:
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
